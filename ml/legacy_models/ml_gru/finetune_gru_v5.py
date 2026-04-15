"""
版本 V5：三模型 Ensemble + Per-user 偏差修正
==============================================
使用 V2 pretrain 權重（相同架構 hidden=64, Attention）
整合 V2~V4 最有效的技術，加上 Ensemble 與偏差修正。

相對 V4 的改動：

  1. 三模型 Ensemble（Seeds: 42, 123, 777）
     理由：單一模型受隨機初始化影響，預測不穩定。
           多個不同初始化的模型各有不同的「盲點」，
           平均後能相互補償，降低方差（variance）。
           即使個別模型表現相近，Ensemble 通常也能
           穩定降低 2~5% 的誤差。

  2. 最佳技術整合（沿用 V3+V4 的優點）：
     - Log1p 目標轉換（V3 的核心改善 SMAPE）
     - Gaussian Noise Augmentation σ=0.02（V3）
     - LLRD（V4：層級 LR，更細緻的遷移學習）
     - Gradual Unfreezing（V2：先凍再解凍）
     - 混合損失 0.7×Huber + 0.3×MSE（V4）

  3. Per-user 偏差修正（Bias Correction）
     理由：每個用戶的消費模式不同，模型可能系統性地
           對某些用戶偏高或偏低。
           做法：
           - 用驗證集計算每個 user 的中位預測誤差（pred - true）
           - 在測試集預測中，對每個 user 減去其偏差
           - 這是無參數的後處理，不改變模型本身，風險低
           注意：這不是 data leakage，因為偏差是從 val set 估計的，
                 不是從 test set。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import copy

ARTIFACTS_DIR  = "artificats"
INPUT_SIZE     = 7
BATCH_SIZE     = 16
PHASE1_EPOCHS  = 50
PHASE2_EPOCHS  = 120
LR_PHASE1      = 3e-4
LR_PHASE2_FC   = 1e-4
LR_PHASE2_ATTN = 3e-5
LR_PHASE2_L1   = 1e-5
LR_PHASE2_L0   = 5e-6
PATIENCE       = 25
WEIGHT_DECAY   = 5e-4
HUBER_DELTA    = 1.0
NOISE_STD      = 0.02
ENSEMBLE_SEEDS = [42, 123, 777]


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

# ─────────────────────────────────────────
# Log1p 目標轉換（沿用 V3 的技術）
# ─────────────────────────────────────────
print("\n📂 載入資料並進行 Log1p 轉換...")

X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_test  = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test  = np.load(f"{ARTIFACTS_DIR}/personal_y_test.npy")
val_user_ids  = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",  allow_pickle=True)
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy", allow_pickle=True)

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    orig_target_scaler = pickle.load(f)

y_train_real = np.clip(orig_target_scaler.inverse_transform(y_train), 0, None)
y_val_real   = np.clip(orig_target_scaler.inverse_transform(y_val),   0, None)

y_train_log  = np.log1p(y_train_real)
y_val_log    = np.log1p(y_val_real)

log_scaler   = StandardScaler()
y_train_sc   = log_scaler.fit_transform(y_train_log)
y_val_sc     = log_scaler.transform(y_val_log)

with open(f"{ARTIFACTS_DIR}/log_target_scaler_v5.pkl", "wb") as f:
    pickle.dump(log_scaler, f)

print(f"  Log1p 轉換完成。Train y 範圍：[{y_train_log.min():.3f}, {y_train_log.max():.3f}]")


class NoisyDataLoader:
    def __init__(self, X_t, y_t, batch_size, noise_std=0.02, shuffle=True):
        self.loader     = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)
        self.noise_std  = noise_std
        self.add_noise  = noise_std > 0

    def __iter__(self):
        for X_b, y_b in self.loader:
            if self.add_noise:
                X_b = X_b + torch.randn_like(X_b) * self.noise_std
            yield X_b, y_b

    def __len__(self):
        return len(self.loader)


class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)
        self.relu       = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


def train_single_model(seed: int, hp: dict, pretrain_state: dict) -> dict:
    """訓練一個模型（給定 seed），回傳最佳 state_dict"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.tensor(X_train,   dtype=torch.float32)
    y_train_t = torch.tensor(y_train_sc, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,     dtype=torch.float32)
    y_val_t   = torch.tensor(y_val_sc,  dtype=torch.float32)

    t_loader = NoisyDataLoader(X_train_t, y_train_t, BATCH_SIZE, noise_std=NOISE_STD)
    v_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    m = GRUWithAttention(INPUT_SIZE, hp["hidden_size"], hp["num_layers"],
                         hp["output_size"], hp["dropout"]).to(device)
    m.load_state_dict(pretrain_state)

    huber_fn = nn.HuberLoss(delta=HUBER_DELTA)
    mse_fn   = nn.MSELoss()

    def loss_fn(p, t):
        return 0.7 * huber_fn(p, t) + 0.3 * mse_fn(p, t)

    gru_params = ["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
                  "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1"]

    def set_phase_freeze(freeze_gru):
        for name, param in m.named_parameters():
            param.requires_grad = not (freeze_gru and name in gru_params)

    # Phase 1
    set_phase_freeze(True)
    opt_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, m.parameters()),
        lr=LR_PHASE1, weight_decay=WEIGHT_DECAY
    )
    sch_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p1, T_max=PHASE1_EPOCHS, eta_min=1e-6)

    best_val  = float("inf")
    best_state = None
    counter    = 0
    phase1_stop = PHASE1_EPOCHS

    print(f"  [Seed {seed}] Phase 1 (凍結GRU)...")
    for epoch in range(1, PHASE1_EPOCHS + 1):
        m.train()
        for X_b, y_b in t_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt_p1.zero_grad()
            loss_fn(m(X_b), y_b).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt_p1.step()

        m.eval()
        vl = 0.0
        with torch.no_grad():
            for X_b, y_b in v_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                vl += loss_fn(m(X_b), y_b).item()
        vl /= len(v_loader)
        sch_p1.step()

        if vl < best_val:
            best_val   = vl
            best_state = copy.deepcopy(m.state_dict())
            counter    = 0
            print(f"    P1 Epoch {epoch:3d} | Val: {vl:.6f} ✅")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"    P1 Early stop at epoch {epoch}")
                phase1_stop = epoch
                break

    # Phase 2：LLRD
    m.load_state_dict(best_state)
    set_phase_freeze(False)

    pg = [
        {"params": [p for n, p in m.named_parameters() if "weight_ih_l0" in n or "weight_hh_l0" in n or "bias_ih_l0" in n or "bias_hh_l0" in n], "lr": LR_PHASE2_L0},
        {"params": [p for n, p in m.named_parameters() if "weight_ih_l1" in n or "weight_hh_l1" in n or "bias_ih_l1" in n or "bias_hh_l1" in n], "lr": LR_PHASE2_L1},
        {"params": [p for n, p in m.named_parameters() if "attention" in n or "layer_norm" in n], "lr": LR_PHASE2_ATTN},
        {"params": [p for n, p in m.named_parameters() if "fc1" in n or "fc2" in n], "lr": LR_PHASE2_FC},
    ]
    opt_p2 = torch.optim.AdamW(pg, weight_decay=WEIGHT_DECAY)
    sch_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_p2, T_0=40, T_mult=2, eta_min=1e-7)

    counter = 0
    print(f"  [Seed {seed}] Phase 2 (LLRD, 全量微調)...")
    for epoch in range(1, PHASE2_EPOCHS + 1):
        m.train()
        for X_b, y_b in t_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt_p2.zero_grad()
            loss_fn(m(X_b), y_b).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt_p2.step()

        m.eval()
        vl = 0.0
        with torch.no_grad():
            for X_b, y_b in v_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                vl += loss_fn(m(X_b), y_b).item()
        vl /= len(v_loader)
        sch_p2.step()

        if vl < best_val:
            best_val   = vl
            best_state = copy.deepcopy(m.state_dict())
            counter    = 0
            print(f"    P2 Epoch {epoch:3d} | Val: {vl:.6f} ✅ (total best)")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"    P2 Early stop at epoch {epoch}")
                break

    print(f"  [Seed {seed}] 完成，最佳 val_loss={best_val:.6f}")
    return best_state, best_val


# ─────────────────────────────────────────
# 訓練三個模型
# ─────────────────────────────────────────
print("\n📦 載入 V2 pretrain 權重...")
checkpoint    = torch.load(f"{ARTIFACTS_DIR}/pretrain_gru_v2.pth", map_location=device)
hp            = checkpoint["hyperparams"]
pretrain_dict = checkpoint["model_state"]
print(f"  ✅ hidden={hp['hidden_size']}")

all_states = []
all_best_vals = []

for seed in ENSEMBLE_SEEDS:
    print(f"\n{'='*60}")
    print(f"  訓練 Ensemble 模型 Seed={seed}...")
    print(f"{'='*60}")
    state, best_val = train_single_model(seed, hp, pretrain_dict)
    all_states.append(state)
    all_best_vals.append(best_val)
    torch.save({
        "model_state": state, "val_loss": best_val,
        "version": "v5", "seed": seed,
        "hyperparams": {"input_size": INPUT_SIZE, "hidden_size": hp["hidden_size"],
                        "num_layers": hp["num_layers"], "output_size": hp["output_size"],
                        "dropout": hp["dropout"]}
    }, f"{ARTIFACTS_DIR}/finetune_gru_v5_seed{seed}.pth")

print(f"\n✅ Ensemble 訓練完成！")
for i, (s, v) in enumerate(zip(ENSEMBLE_SEEDS, all_best_vals)):
    print(f"  Seed {s} : best val_loss = {v:.6f}")

print("\n下一步：執行 predict_v5.py")
