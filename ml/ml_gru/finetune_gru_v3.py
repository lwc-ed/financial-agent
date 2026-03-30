"""
版本 V3：GRU Finetune（Log1p 目標轉換 + 輸入噪聲增強）
==========================================================
使用 V2 pretrain 權重（相同架構 hidden=64, Attention）

相對 V2 的改動：

  1. Log1p 目標轉換（最核心改動）
     理由：個人消費金額分布高度偏態（right-skewed）：
           大多數日子花費較少，偶爾有大額消費。
           在原始金額空間訓練時，模型被大額消費主導，
           導致對小額消費的預測誤差比例偏高（SMAPE 仍高達 78%）。
           Log1p 空間讓大小消費的預測難度更均衡，
           理論上直接改善 SMAPE 和 Per-user NMAE。

     做法：
     - 把 y_train/y_val 從標準化空間反轉回實際金額
     - 套用 log1p 轉換（log(1+x)，因 x≥0 後 clipping）
     - 再用新的 StandardScaler 標準化
     - 預測時：反標準化 → expm1 → 實際金額

  2. 輸入噪聲增強（Gaussian Noise Augmentation）
     理由：16 個用戶、4067 筆訓練資料不足以讓模型泛化。
           在輸入序列加入少量高斯噪聲（σ=0.02），
           等同於人工增加訓練樣本的多樣性，
           防止模型記憶特定的輸入模式。
           噪聲只在訓練時加入，驗證/測試不加。

  3. 凍結策略回歸 V0：Phase 1 凍結 GRU（沿用 V2 的 Gradual Unfreezing）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle

ARTIFACTS_DIR   = "artificats"
INPUT_SIZE      = 7
BATCH_SIZE      = 16
PHASE1_EPOCHS   = 60
PHASE2_EPOCHS   = 140
LR_PHASE1       = 3e-4
LR_PHASE2       = 1e-4
PATIENCE        = 25
WEIGHT_DECAY    = 5e-4
HUBER_DELTA     = 1.0
NOISE_STD       = 0.02   # V3 新增：輸入噪聲強度

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

print("\n📂 載入個人記帳資料並進行 Log1p 轉換...")

X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")

# 載入原始 target_scaler（StandardScaler in original space）
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    orig_target_scaler = pickle.load(f)

# 反轉換回實際金額（winsorized 但未 log 轉換）
y_train_real = orig_target_scaler.inverse_transform(y_train)
y_val_real   = orig_target_scaler.inverse_transform(y_val)

# 確保非負（clipping 後應已非負，但以防萬一）
y_train_real = np.clip(y_train_real, 0, None)
y_val_real   = np.clip(y_val_real,   0, None)

# Log1p 轉換
y_train_log = np.log1p(y_train_real)
y_val_log   = np.log1p(y_val_real)

print(f"  y_train 實際值範圍   : [{y_train_real.min():.1f}, {y_train_real.max():.1f}]")
print(f"  y_train log1p 範圍  : [{y_train_log.min():.3f}, {y_train_log.max():.3f}]")
print(f"  原始標準差：{y_train_real.std():.2f}  Log1p 標準差：{y_train_log.std():.3f}")

# 在 log1p 空間重新標準化（fit on train only）
log_target_scaler = StandardScaler()
y_train_scaled = log_target_scaler.fit_transform(y_train_log)
y_val_scaled   = log_target_scaler.transform(y_val_log)

print(f"  Log1p 標準化後 train 均值：{y_train_scaled.mean():.4f}，標準差：{y_train_scaled.std():.4f}")

# 儲存 log_target_scaler 供 predict_v3.py 使用
with open(f"{ARTIFACTS_DIR}/log_target_scaler_v3.pkl", "wb") as f:
    pickle.dump(log_target_scaler, f)

X_train_t = torch.tensor(X_train,        dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,          dtype=torch.float32)
y_val_t   = torch.tensor(y_val_scaled,   dtype=torch.float32)

print(f"\n  X_train : {X_train.shape}")
print(f"  y_train (log1p scaled) : {y_train_scaled.shape}")


class NoisyDataLoader:
    """
    訓練時在輸入 X 加入高斯噪聲，驗證時不加。
    理由：資料增強，防止過擬合。σ=0.02 是標準化後的尺度（約 2% 標準差）。
    """
    def __init__(self, X_t, y_t, batch_size, noise_std=0.02, shuffle=True):
        self.dataset    = TensorDataset(X_t, y_t)
        self.loader     = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.noise_std  = noise_std
        self.add_noise  = noise_std > 0

    def __iter__(self):
        for X_b, y_b in self.loader:
            if self.add_noise:
                X_b = X_b + torch.randn_like(X_b) * self.noise_std
            yield X_b, y_b

    def __len__(self):
        return len(self.loader)


train_loader = NoisyDataLoader(X_train_t, y_train_t, BATCH_SIZE, noise_std=NOISE_STD, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)


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


print("\n📦 載入 V2 pretrain 權重（V3 複用 V2 架構）...")
checkpoint = torch.load(f"{ARTIFACTS_DIR}/pretrain_gru_v2.pth", map_location=device)
hp         = checkpoint["hyperparams"]

model = GRUWithAttention(
    INPUT_SIZE, hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
model.load_state_dict(checkpoint["model_state"])
print(f"  ✅ 載入完成（hidden={hp['hidden_size']}）")


def set_phase(phase):
    gru_params = ["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
                  "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1"]
    for name, param in model.named_parameters():
        param.requires_grad = not (phase == 1 and name in gru_params)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Phase {phase}：可訓練 {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")


criterion     = nn.HuberLoss(delta=HUBER_DELTA)
best_val_loss = float("inf")
best_model_path = f"{ARTIFACTS_DIR}/finetune_gru_v3.pth"
train_losses, val_losses = [], []

# ─── Phase 1：凍結 GRU ───
print("\n🔒 V3 Phase 1：凍結 GRU，Log1p + Noise 訓練 Attention+FC...")
set_phase(1)
optimizer_p1 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE1, weight_decay=WEIGHT_DECAY
)
scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=PHASE1_EPOCHS, eta_min=1e-6)

patience_counter = 0
phase1_stop      = PHASE1_EPOCHS
print(f"\n🚀 Phase 1（{PHASE1_EPOCHS} epochs max）...")
print("=" * 65)

for epoch in range(1, PHASE1_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer_p1.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_p1.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_loss += criterion(model(X_b), y_b).item()
    val_loss /= len(val_loader)

    scheduler_p1.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    lr = optimizer_p1.param_groups[0]["lr"]
    print(f"  [P1] Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch": epoch, "phase": 1, "model_state": model.state_dict(),
            "val_loss": best_val_loss, "version": "v3",
            "hyperparams": {"input_size": INPUT_SIZE, "hidden_size": hp["hidden_size"],
                            "num_layers": hp["num_layers"], "output_size": hp["output_size"],
                            "dropout": hp["dropout"]}
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Phase 1 Early stopping at epoch {epoch}！")
            phase1_stop = epoch
            break

# ─── Phase 2：全量微調 ───
print(f"\n🔓 V3 Phase 2：解凍所有層...")
set_phase(2)
optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-7)

patience_counter = 0
print(f"\n🚀 Phase 2（{PHASE2_EPOCHS} epochs max）...")
print("=" * 65)

for epoch in range(1, PHASE2_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer_p2.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_p2.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_loss += criterion(model(X_b), y_b).item()
    val_loss /= len(val_loader)

    scheduler_p2.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    lr = optimizer_p2.param_groups[0]["lr"]
    print(f"  [P2] Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch": phase1_stop + epoch, "phase": 2, "model_state": model.state_dict(),
            "val_loss": best_val_loss, "version": "v3",
            "hyperparams": {"input_size": INPUT_SIZE, "hidden_size": hp["hidden_size"],
                            "num_layers": hp["num_layers"], "output_size": hp["output_size"],
                            "dropout": hp["dropout"]}
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Phase 2 Early stopping！")
            break

print("=" * 65)
print(f"\n🎉 V3 Finetune 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/finetune_history_v3.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 predict_v3.py")
