"""
版本 V4：GRU Finetune（Layer-wise LR Decay + 混合損失）
==========================================================
使用 V2 pretrain 權重（相同架構 hidden=64, Attention）

相對 V3 的改動（回到原始空間，不用 log1p，評估 LLRD 的純效果）：

  1. Layer-wise Learning Rate Decay（LLRD）
     理由：遷移學習中，距離 pretrain 越遠的層（越靠近輸出）
           需要更大的 LR 快速適應新任務；
           越靠近輸入的層（GRU 底層）已學到通用序列特徵，
           用更小的 LR 微調以避免破壞 pretrain 知識。
           這比「全凍結→全解凍」更細緻，讓每層以最適合的速度調整。

     LR 分組（由底到頂）：
       GRU Layer 0（最底層）：LR = 5e-6（最小，最不動）
       GRU Layer 1：          LR = 1e-5
       Attention + LayerNorm：LR = 3e-5（中間）
       FC1 + FC2（最頂層）：  LR = 1e-4（最大，最快適應）

  2. 混合損失函數：0.7 × HuberLoss + 0.3 × MSELoss
     理由：
       - HuberLoss 對離群值穩健，改善 SMAPE
       - MSELoss 對所有誤差均等懲罰，維持 RMSE 不要退步
       - 混合兩者試圖在 SMAPE 和 MAE/RMSE 間取得平衡
       - V1 切換到純 HuberLoss 後 RMSE 退步，
         V4 用混合損失希望兩者都好

  3. CosineAnnealingWarmRestarts（T_0=40, T_mult=2）
     理由：Warm Restarts 讓 LR 週期性上升，
           幫助模型跳出局部最優，探索更好的解空間。
           比 fixed cosine annealing 更有機會找到更低的 val loss。

  4. 無噪聲增強（以便和 V3 做公平比較）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle

ARTIFACTS_DIR = "artificats"
INPUT_SIZE    = 7
BATCH_SIZE    = 16
EPOCHS        = 200
PATIENCE      = 30
WEIGHT_DECAY  = 5e-4
HUBER_DELTA   = 1.0
HUBER_WEIGHT  = 0.7   # 混合損失權重
MSE_WEIGHT    = 0.3

# LLRD：由頂層到底層 LR 遞減
LR_FC         = 1e-4    # fc1, fc2（最頂層）
LR_ATTN       = 3e-5    # attention, layer_norm
LR_GRU_L1    = 1e-5    # GRU layer 1
LR_GRU_L0    = 5e-6    # GRU layer 0（最底層）

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

print("\n📂 載入個人記帳資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
print(f"  X_train : {X_train.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)


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


print("\n📦 載入 V2 pretrain 權重（V4 複用 V2 架構）...")
checkpoint = torch.load(f"{ARTIFACTS_DIR}/pretrain_gru_v2.pth", map_location=device)
hp         = checkpoint["hyperparams"]

model = GRUWithAttention(
    INPUT_SIZE, hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
model.load_state_dict(checkpoint["model_state"])
print(f"  ✅ 載入完成（hidden={hp['hidden_size']}）")


# ─────────────────────────────────────────
# LLRD：為不同層組設定不同 LR
# ─────────────────────────────────────────
print(f"\n📐 Layer-wise LR Decay 設定：")
print(f"  GRU Layer 0  : LR = {LR_GRU_L0:.0e}")
print(f"  GRU Layer 1  : LR = {LR_GRU_L1:.0e}")
print(f"  Attention    : LR = {LR_ATTN:.0e}")
print(f"  FC Head      : LR = {LR_FC:.0e}")

param_groups = [
    {"params": [p for n, p in model.named_parameters() if "weight_ih_l0" in n or "weight_hh_l0" in n or "bias_ih_l0" in n or "bias_hh_l0" in n],
     "lr": LR_GRU_L0, "name": "gru_layer0"},
    {"params": [p for n, p in model.named_parameters() if "weight_ih_l1" in n or "weight_hh_l1" in n or "bias_ih_l1" in n or "bias_hh_l1" in n],
     "lr": LR_GRU_L1, "name": "gru_layer1"},
    {"params": [p for n, p in model.named_parameters() if "attention" in n or "layer_norm" in n],
     "lr": LR_ATTN, "name": "attention"},
    {"params": [p for n, p in model.named_parameters() if "fc1" in n or "fc2" in n],
     "lr": LR_FC, "name": "fc_head"},
]

# 混合損失函數
huber_criterion = nn.HuberLoss(delta=HUBER_DELTA)
mse_criterion   = nn.MSELoss()


def mixed_loss(pred, target):
    return HUBER_WEIGHT * huber_criterion(pred, target) + MSE_WEIGHT * mse_criterion(pred, target)


optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
# CosineAnnealingWarmRestarts：T_0=40 表示第一個重啟週期 40 epochs，之後每次 ×2
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=40, T_mult=2, eta_min=1e-7
)

print(f"\n  損失函數：{HUBER_WEIGHT}×Huber + {MSE_WEIGHT}×MSE")
print(f"  排程器：CosineAnnealingWarmRestarts (T_0=40, T_mult=2)")

print("\n🚀 開始 V4 Finetune（LLRD + 混合損失 + WarmRestarts）...")
print("=" * 65)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/finetune_gru_v4.pth"
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = mixed_loss(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_loss += mixed_loss(model(X_b), y_b).item()
    val_loss /= len(val_loader)

    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr_fc = optimizer.param_groups[3]["lr"]
    print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR(FC): {lr_fc:.2e}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "val_loss": best_val_loss, "version": "v4",
            "hyperparams": {"input_size": INPUT_SIZE, "hidden_size": hp["hidden_size"],
                            "num_layers": hp["num_layers"], "output_size": hp["output_size"],
                            "dropout": hp["dropout"]}
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping！")
            break

print("=" * 65)
print(f"\n🎉 V4 Finetune 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/finetune_history_v4.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 predict_v4.py")
