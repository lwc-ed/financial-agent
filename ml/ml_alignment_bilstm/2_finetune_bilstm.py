"""
Step 2：Bi-LSTM Global Finetune（個人資料）
==========================================
載入 Pretrained Bi-LSTM，在全體個人 aligned 資料上做 finetune
7-seed ensemble，不做 per-user finetune（先做到這步）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from model_bilstm import BiLSTMWithAttention

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
SRC_ARTIFACTS  = "../ml_alignment_lwc/artifacts_aligned"
SAVE_DIR       = "artifacts_bilstm"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 超參數 ────────────────────────────────────────────────────────────────────
INPUT_SIZE    = 7
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 32
EPOCHS        = 80
LEARNING_RATE = 3e-4
PATIENCE      = 20
WEIGHT_DECAY  = 1e-4
SEEDS         = [42, 123, 777, 456, 789, 999, 2024]

# ── 裝置 ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── 載入個人 aligned 資料 ─────────────────────────────────────────────────────
print("📂 載入個人 aligned 資料...")
X_train = np.load(f"{SRC_ARTIFACTS}/personal_aligned_X_train.npy")
y_train = np.load(f"{SRC_ARTIFACTS}/personal_aligned_y_train.npy")
X_val   = np.load(f"{SRC_ARTIFACTS}/personal_aligned_X_val.npy")
y_val   = np.load(f"{SRC_ARTIFACTS}/personal_aligned_y_val.npy")
print(f"  X_train : {X_train.shape}  X_val : {X_val.shape}")

val_t = torch.tensor(X_val).to(device)
val_y = torch.tensor(y_val).to(device)
criterion = nn.HuberLoss(delta=1.0)

# ── Pretrain checkpoint ───────────────────────────────────────────────────────
pretrain_path = f"{SAVE_DIR}/pretrain_bilstm.pth"
if not os.path.exists(pretrain_path):
    print("❌ 找不到 pretrain checkpoint，請先執行 1_pretrain_bilstm.py")
    exit(1)

# ── 7-seed Ensemble Finetune ──────────────────────────────────────────────────
print(f"\n🚀 Ensemble Finetune（seeds={SEEDS}）...")
print("=" * 60)

for seed in SEEDS:
    print(f"\n  ── Seed {seed} ──")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 載入 pretrain weights
    ckpt  = torch.load(pretrain_path, map_location=device)
    model = BiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_t), val_y).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    save_path = f"{SAVE_DIR}/finetune_bilstm_seed{seed}.pth"
    torch.save({"model_state": best_state, "seed": seed, "val_loss": best_val_loss}, save_path)
    print(f"    ✅ best_val_loss={best_val_loss:.6f}  →  {save_path}")

print("\n🎉 Finetune 完成！")
print("下一步：執行 3_predict_bilstm.py")
