"""
Step 4：BiLSTM Aligned Finetune
=================================
載入 Aligned Pretrained BiLSTM，在個人 Aligned 資料上 finetune
Loss = HuberLoss + λ * MMD_loss（與 pretrain 保持一致）
  - 兩個 loss 分開印，方便確認 scale 與 pretrain 是否一致
Ensemble 多 seeds，讓暴力搜尋找到最佳組合
輸出：
  - finetune_bilstm_seed{seed}.pth
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_bilstm_v2"

# ── 超參數 ────────────────────────────────────────────────────────────────────
INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)   # 10
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 32
EPOCHS        = 80
LEARNING_RATE = 3e-4
PATIENCE      = 20
WEIGHT_DECAY  = 1e-4
HUBER_DELTA   = 1.0
MMD_LAMBDA    = 0.1    # ← 與 pretrain 保持一致，看 scale 後再調

SEEDS = [42, 123, 777, 456, 789, 999, 2024,
         0, 7, 13, 21, 100, 314, 1234, 9999]

# ── 設備 ──────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps");  print("✅ Apple M1 MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("✅ CUDA")
else:
    device = torch.device("cpu");  print("⚠️  CPU")

# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入個人 Aligned 資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

# Walmart 資料（只用 X，不用 label）── 給 MMD 計算用
print("📂 載入 Walmart 資料（用於 MMD alignment）...")
X_walmart = np.load(f"{ARTIFACTS_DIR}/walmart_X_train.npy")
print(f"  X_walmart: {X_walmart.shape}")

walmart_loader = DataLoader(
    TensorDataset(torch.tensor(X_walmart, dtype=torch.float32)),
    batch_size=BATCH_SIZE, shuffle=True
)


# ── MMD 計算（RBF kernel）─────────────────────────────────────────────────────
def compute_mmd(x: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """
    Maximum Mean Discrepancy（RBF kernel）
    x: source domain representations  (B1, D)
    y: target domain representations  (B2, D)
    """
    n = x.size(0)
    m = y.size(0)

    rx = (x ** 2).sum(dim=1, keepdim=True)   # (n, 1)
    ry = (y ** 2).sum(dim=1, keepdim=True)   # (m, 1)

    dist_xx = rx + rx.t() - 2 * torch.mm(x, x.t())   # (n, n)
    K = torch.exp(-0.5 * dist_xx / bandwidth ** 2)

    dist_yy = ry + ry.t() - 2 * torch.mm(y, y.t())   # (m, m)
    L = torch.exp(-0.5 * dist_yy / bandwidth ** 2)

    dist_xy = rx + ry.t() - 2 * torch.mm(x, y.t())   # (n, m)
    P = torch.exp(-0.5 * dist_xy / bandwidth ** 2)

    mmd = (K.sum() - K.trace()) / (n * (n - 1) + 1e-8) \
        + (L.sum() - L.trace()) / (m * (m - 1) + 1e-8) \
        - 2 * P.mean()
    return mmd.clamp(min=0)


# ── 模型架構 ──────────────────────────────────────────────────────────────────
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        bi_hidden = hidden_size * 2
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bi_hidden, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        """回傳 attended hidden representation（用於 MMD）"""
        out, _ = self.lstm(x)
        attn_w  = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


def load_pretrained():
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/pretrain_bilstm.pth", map_location=device)
    model = BiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model


# ── Ensemble Finetune ─────────────────────────────────────────────────────────
print(f"\n🚀 Ensemble Finetune（{len(SEEDS)} seeds，MMD_LAMBDA={MMD_LAMBDA}）...")
print(f"   seeds = {SEEDS}")

for seed in SEEDS:
    save_path = f"{ARTIFACTS_DIR}/finetune_bilstm_seed{seed}.pth"

    if os.path.exists(save_path):
        print(f"\n  Seed {seed}：已存在，跳過")
        continue

    print(f"\n{'='*70}")
    print(f"  Seed {seed}")
    print(f"{'='*70}")
    print(f"  {'Epoch':>5}  {'HuberLoss':>10}  {'MMD_loss':>10}  {'Total':>10}  {'Val':>10}  {'LR':>8}")
    print(f"  {'-'*65}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = load_pretrained().to(device)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val),   torch.tensor(y_val)),
                              batch_size=BATCH_SIZE, shuffle=False)

    walmart_iter     = iter(walmart_loader)
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_huber = 0.0
        epoch_mmd   = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            # 取一個 Walmart batch（循環用）
            try:
                (X_w,) = next(walmart_iter)
            except StopIteration:
                walmart_iter = iter(walmart_loader)
                (X_w,) = next(walmart_iter)
            X_w = X_w.to(device)

            optimizer.zero_grad()

            # Task loss（Huber）
            huber_loss = criterion(model(X_b), y_b)

            # MMD loss：個人 hidden rep vs Walmart hidden rep
            rep_personal = model.encode(X_b)
            rep_walmart  = model.encode(X_w)
            mmd_loss     = compute_mmd(rep_personal, rep_walmart)

            total_loss = huber_loss + MMD_LAMBDA * mmd_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_huber += huber_loss.item()
            epoch_mmd   += mmd_loss.item()

        epoch_huber /= len(train_loader)
        epoch_mmd   /= len(train_loader)
        epoch_total  = epoch_huber + MMD_LAMBDA * epoch_mmd

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  {epoch:5d}  {epoch_huber:10.6f}  {epoch_mmd:10.6f}  {epoch_total:10.6f}  {val_loss:10.6f}  {lr:8.2e}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : best_val_loss,
                "seed"       : seed,
                "version"    : "bilstm_finetune_mmd",
                "mmd_lambda" : MMD_LAMBDA,
            }, save_path)
            print(f"             ✅ 儲存（val_loss={best_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹️  Early stopping")
                break

    print(f"  Seed {seed} 最佳 Val Loss: {best_val_loss:.6f}")

print(f"\n🎉 Finetune 完成！")
print(f"   💡 觀察各 seed 的 HuberLoss 和 MMD_loss scale，若差距太大請調整 MMD_LAMBDA（目前={MMD_LAMBDA}）")
print(f"   → 下一步：5_predict_bilstm.py")
