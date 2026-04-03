"""
GRU 無 Pretrain 版本（Ensemble）
=================================
完全不使用 Walmart 預訓練權重，直接用個人資料從頭訓練。

目的：對比是否需要 pretrain，驗證 transfer learning 的實際效益。

技術設定（沿用 V5 最佳實踐）：
  - Log1p 目標轉換
  - Gaussian Noise Augmentation σ=0.02
  - 三模型 Ensemble（Seeds: 42, 123, 777）
  - 混合損失 0.7×Huber + 0.3×MSE
  - CosineAnnealingWarmRestarts
  - Per-user 偏差修正（在 predict_nopretrain.py 處理）

差異（相對 V5）：
  - 不載入任何 pretrain 權重，全部隨機初始化
  - 無 Phase 1/2 分階段訓練，直接端到端訓練
  - 訓練 epochs 增加（彌補沒有 pretrain 熱身的不足）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import copy

ARTIFACTS_DIR  = "artificats"
VERSION        = "nopretrain"
INPUT_SIZE     = 7
HIDDEN_SIZE    = 64
NUM_LAYERS     = 2
OUTPUT_SIZE    = 1
DROPOUT        = 0.4
BATCH_SIZE     = 16
EPOCHS         = 300
PATIENCE       = 40
LR             = 1e-3
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
# Log1p 目標轉換
# ─────────────────────────────────────────
print("\n📂 載入資料並進行 Log1p 轉換...")

X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    orig_target_scaler = pickle.load(f)

y_train_real = np.clip(orig_target_scaler.inverse_transform(y_train), 0, None)
y_val_real   = np.clip(orig_target_scaler.inverse_transform(y_val),   0, None)

y_train_log  = np.log1p(y_train_real)
y_val_log    = np.log1p(y_val_real)

log_scaler   = StandardScaler()
y_train_sc   = log_scaler.fit_transform(y_train_log)
y_val_sc     = log_scaler.transform(y_val_log)

with open(f"{ARTIFACTS_DIR}/log_target_scaler_{VERSION}.pkl", "wb") as f:
    pickle.dump(log_scaler, f)

print(f"  Log1p 轉換完成。Train y 範圍：[{y_train_log.min():.3f}, {y_train_log.max():.3f}]")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")


class NoisyDataLoader:
    def __init__(self, X_t, y_t, batch_size, noise_std=0.02, shuffle=True):
        self.loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)
        self.noise_std = noise_std

    def __iter__(self):
        for X_b, y_b in self.loader:
            if self.noise_std > 0:
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


def train_single_model(seed: int) -> tuple:
    """訓練一個模型（從頭，無 pretrain）"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.tensor(X_train,   dtype=torch.float32)
    y_train_t = torch.tensor(y_train_sc, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,     dtype=torch.float32)
    y_val_t   = torch.tensor(y_val_sc,  dtype=torch.float32)

    t_loader = NoisyDataLoader(X_train_t, y_train_t, BATCH_SIZE, noise_std=NOISE_STD)
    v_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [Seed {seed}] 參數量: {total_params:,}  |  完全隨機初始化，無 pretrain")

    huber_fn = nn.HuberLoss(delta=HUBER_DELTA)
    mse_fn   = nn.MSELoss()

    def loss_fn(p, t):
        return 0.7 * huber_fn(p, t) + 0.3 * mse_fn(p, t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=40, T_mult=2, eta_min=1e-7)

    best_val   = float("inf")
    best_state = None
    counter    = 0

    print(f"  [Seed {seed}] 開始訓練（最多 {EPOCHS} epochs, patience={PATIENCE}）...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in t_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss_fn(model(X_b), y_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X_b, y_b in v_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                vl += loss_fn(model(X_b), y_b).item()
        vl /= len(v_loader)
        scheduler.step()

        if vl < best_val:
            best_val   = vl
            best_state = copy.deepcopy(model.state_dict())
            counter    = 0
            print(f"    Epoch {epoch:3d} | Val: {vl:.6f} ✅")
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    print(f"  [Seed {seed}] 完成，最佳 val_loss={best_val:.6f}")
    return best_state, best_val


# ─────────────────────────────────────────
# 訓練三個 Ensemble 模型
# ─────────────────────────────────────────
print(f"\n🚀 開始 Ensemble 訓練（{len(ENSEMBLE_SEEDS)} seeds，無 pretrain）...")

HP = {
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "output_size": OUTPUT_SIZE,
    "dropout": DROPOUT,
}

all_best_vals = []
for seed in ENSEMBLE_SEEDS:
    print(f"\n{'='*60}")
    print(f"  Ensemble 模型 Seed={seed}（無 pretrain）")
    print(f"{'='*60}")
    state, best_val = train_single_model(seed)
    all_best_vals.append(best_val)
    torch.save({
        "model_state": state,
        "val_loss": best_val,
        "version": VERSION,
        "seed": seed,
        "pretrained": False,
        "hyperparams": HP,
    }, f"{ARTIFACTS_DIR}/gru_{VERSION}_seed{seed}.pth")

print(f"\n✅ 全部 Ensemble 模型訓練完成！")
for seed, val in zip(ENSEMBLE_SEEDS, all_best_vals):
    print(f"  Seed {seed} : best val_loss = {val:.6f}")

print(f"\n下一步：執行 predict_nopretrain.py")
