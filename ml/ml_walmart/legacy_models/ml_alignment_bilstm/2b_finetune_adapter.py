"""
Step 2b：Adapter Finetune
==========================
策略：兩階段訓練
  Phase 1（前 N epochs）：凍結 BiLSTM body，只訓練 Adapter + FC
                          → 讓 Adapter 先學會把 raw features 投影到 pretrain 空間
  Phase 2（之後）        ：解凍所有層，用低 LR 一起 finetune
                          → 精細調整整個模型

這樣可以避免 Adapter 的隨機梯度在初期破壞 BiLSTM body 的 pretrain weights
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from model_bilstm_adapter import BiLSTMWithAdapter

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
SAVE_DIR      = "artifacts_bilstm"
PRETRAIN_PATH = f"{SAVE_DIR}/pretrain_bilstm.pth"

# ── 超參數 ────────────────────────────────────────────────────────────────────
RAW_INPUT_SIZE    = 5
PRETRAIN_INPUT    = 7
HIDDEN_SIZE       = 64
NUM_LAYERS        = 2
DROPOUT           = 0.4
BATCH_SIZE        = 32
PHASE1_EPOCHS     = 20       # 只訓練 Adapter + FC
PHASE2_EPOCHS     = 60       # 全部解凍
PHASE1_LR         = 1e-3     # Adapter 學習率（高一點，因為從頭學）
PHASE2_LR         = 5e-5     # 全局學習率（低，保護 pretrain weights）
PATIENCE          = 15
WEIGHT_DECAY      = 1e-4
SEEDS             = [42, 123, 777, 456, 789, 999, 2024]

# ── 裝置 ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"裝置：{device}")

# ── 載入 raw feature 個人資料 ─────────────────────────────────────────────────
print("\n📂 載入 raw feature 個人資料...")
X_train  = np.load(f"{SAVE_DIR}/raw_X_train.npy")
y_train  = np.load(f"{SAVE_DIR}/raw_y_train_s.npy")
X_val    = np.load(f"{SAVE_DIR}/raw_X_val.npy")
y_val    = np.load(f"{SAVE_DIR}/raw_y_val_s.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

val_t = torch.tensor(X_val).to(device)
val_y = torch.tensor(y_val).to(device)
criterion = nn.HuberLoss(delta=1.0)

# ── 7-seed Ensemble Finetune ──────────────────────────────────────────────────
print(f"\n🚀 Adapter Ensemble Finetune（seeds={SEEDS}）...")
print("=" * 60)

for seed in SEEDS:
    print(f"\n  ── Seed {seed} ──")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 建立模型並載入 pretrain body weights
    model = BiLSTMWithAdapter(
        RAW_INPUT_SIZE, PRETRAIN_INPUT, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT
    ).to(device)
    model.load_pretrained_body(PRETRAIN_PATH, device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    # ── Phase 1：凍結 BiLSTM，只訓練 Adapter + FC ─────────────────────────
    print(f"    Phase 1（凍結 BiLSTM，訓練 Adapter+FC，{PHASE1_EPOCHS} epochs）")
    for param in model.lstm.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    for epoch in range(1, PHASE1_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(val_t), val_y).item()
        scheduler.step(vl)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"    Phase 1 完成，best_val_loss={best_val_loss:.6f}")

    # ── Phase 2：解凍所有層，低 LR 全局 finetune ──────────────────────────
    print(f"    Phase 2（全部解凍，LR={PHASE2_LR}，最多 {PHASE2_EPOCHS} epochs）")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )
    patience_cnt = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(val_t), val_y).item()
        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            patience_cnt  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stopping at Phase 2 epoch {epoch}")
                break

    save_path = f"{SAVE_DIR}/adapter_bilstm_seed{seed}.pth"
    torch.save({"model_state": best_state, "seed": seed,
                "val_loss": best_val_loss}, save_path)
    print(f"    ✅ best_val_loss={best_val_loss:.6f}  →  {save_path}")

print("\n🎉 Adapter Finetune 完成！")
print("下一步：執行 3b_predict_adapter.py")
