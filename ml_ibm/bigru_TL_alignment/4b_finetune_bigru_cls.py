import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from imblearn.over_sampling import SMOTE 

# ── 1. 路徑鎖定 ──────────────────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention

# ── 2. 超參數設定 ────────────────────────────────────────────────────────
INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
# 你朋友建議的權重：調輕一點點 [1.0, 2.0, 3.0, 1.0]，避免模型太疑神疑鬼
CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 3.0, 1.0]) 

BATCH_SIZE = 32
EPOCHS = 50 
LEARNING_RATE = 3e-4
PATIENCE = 10
SEEDS = [42, 123, 777, 456, 789, 999, 2024]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️  使用設備: {device} | 任務：風險等級分類 (F1 優化版)")

# ── 3. 載入資料 ──────────────────────────────────────────────────────────
# 【關鍵修改】：直接載入 2b 產出的官方標籤，不再用 amount_to_label 亂切
print(f"📂 載入資料與官方財務標籤...")
X_train = np.load(ARTIFACTS_DIR / "personal_X_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "personal_X_val.npy")

try:
    y_train_label = np.load(ARTIFACTS_DIR / "personal_y_train_label.npy").astype(np.int64)
    y_val_label   = np.load(ARTIFACTS_DIR / "personal_y_val_label.npy").astype(np.int64)
except FileNotFoundError:
    print("❌ 找不到標籤檔！請先執行 python3 2b_generate_labels.py")
    sys.exit()

# ✨ 執行 SMOTE 平衡
print(f"⚖️  執行 SMOTE 平衡中...")
N, T, F = X_train.shape
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train.reshape(N, -1), y_train_label)
X_train_final = X_res.reshape(-1, T, F)
y_train_final = y_res
print(f"✅ 樣本平衡完成：訓練集樣本從 {N} 提升至 {len(X_train_final)}")

# ── 4. 載入預訓練模型 ─────────────────────────────────────────────────────
def load_pretrained_for_classification():
    PRETRAIN_PATH = ARTIFACTS_DIR / "pretrain_bigru.pth"
    # 先建立原本輸出 1 的模型，載入 IBM 大腦
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size=1, dropout=DROPOUT)
    ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    # 關鍵手術：換成輸出 4 類
    model.fc2 = nn.Linear(HIDDEN_SIZE, 4) 
    return model.to(device)

# ── 5. 訓練迴圈 ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))

for seed in SEEDS:
    save_path = ARTIFACTS_DIR / f"finetune_bigru_cls_seed{seed}.pth"
    print(f"🔥 Seed {seed} 訓練中...", end=" ")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = load_pretrained_for_classification()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_final), torch.tensor(y_train_final)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val_label)), batch_size=BATCH_SIZE)

    best_v_loss = float("inf")
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(); loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()

        model.eval(); v_loss = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                v_loss += criterion(model(xv.to(device)), yv.to(device)).item()
        v_loss /= len(val_loader)

        if v_loss < best_v_loss:
            best_v_loss = v_loss; patience_cnt = 0
            torch.save({"model_state": model.state_dict()}, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break
    
    print(f"最佳 Val Loss: {best_v_loss:.4f}")

print("\n🎉 [路線 B] 微調結束！")