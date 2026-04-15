"""
Step 4：個人資料 Fine-tune（分類任務）
========================================
從 Walmart 預訓練的 GRUClassifier 出發，在個人資料上 fine-tune
策略：
  - 載入 pretrain encoder 權重
  - 以較高 LR fine-tune 全部層（個人資料小，直接整體微調）
  - 多 seeds 訓練，記錄每個 seed 的 val F1，供後續 ensemble

輸出：artifacts_clf/finetune_clf_seed{seed}.pth（每個 seed 一個）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, sys, random
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_clf"

INPUT_SIZE   = len(ALIGNED_FEATURE_COLS)   # 12
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3    # 個人資料小，稍微降低 dropout
BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 3e-4
PATIENCE     = 20
WEIGHT_DECAY = 1e-4

SEEDS = [42, 123, 456, 789, 1000, 2024, 7, 13, 21, 314]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"🖥️  Device: {device}")


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, 1)
        self.relu       = nn.ReLU()

    def encode(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入個人分類資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_clf_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_clf_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_clf_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_clf_y_val.npy")
pos_weight_val = float(np.load(f"{ARTIFACTS_DIR}/personal_clf_pos_weight.npy")[0])
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  pos_weight={pos_weight_val:.3f}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=False
)

PRETRAIN_PATH = f"{ARTIFACTS_DIR}/pretrain_clf_gru.pth"
if not os.path.exists(PRETRAIN_PATH):
    raise FileNotFoundError(f"❌ 找不到預訓練模型：{PRETRAIN_PATH}，請先執行 3_pretrain_clf.py")

seed_results = {}
print(f"\n🚀 開始 Fine-tune（{len(SEEDS)} 個 seeds）...")

for seed in SEEDS:
    ckpt_path = f"{ARTIFACTS_DIR}/finetune_clf_seed{seed}.pth"
    if os.path.exists(ckpt_path):
        info = torch.load(ckpt_path, map_location="cpu")
        print(f"  seed={seed:5d}  ✅ 已存在（val_f1={info.get('val_f1', '?'):.4f}），跳過")
        seed_results[seed] = info.get("val_f1", 0.0)
        continue

    # 固定隨機種子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = GRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)

    # 載入 pretrain 權重（只載入 encoder 部分也可以，這裡載全部 + 容許 head mismatch）
    pretrain_state = torch.load(PRETRAIN_PATH, map_location=device)["model_state"]
    # 只載入 shape 吻合的層（head 也一樣所以會全部載入）
    model.load_state_dict(pretrain_state, strict=False)

    pos_w_t   = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1      = -1.0
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits = model(X_b.to(device))
                val_loss += criterion(logits, y_b.to(device)).item()
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                val_preds.extend((probs > 0.5).astype(int).tolist())
                val_labels.extend(y_b.numpy().ravel().astype(int).tolist())
        val_loss /= len(val_loader)

        vp, vl = np.array(val_preds), np.array(val_labels)
        tp = int(np.sum((vp == 1) & (vl == 1)))
        fp = int(np.sum((vp == 1) & (vl == 0)))
        fn = int(np.sum((vp == 0) & (vl == 1)))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        if f1 > best_val_f1 or (f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_f1      = f1
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_f1"      : best_val_f1,
                "val_loss"    : best_val_loss,
                "seed"        : seed,
                "hyperparams" : {
                    "input_size"  : INPUT_SIZE,
                    "hidden_size" : HIDDEN_SIZE,
                    "num_layers"  : NUM_LAYERS,
                    "dropout"     : DROPOUT,
                    "feature_cols": ALIGNED_FEATURE_COLS,
                    "task"        : "classification",
                }
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    seed_results[seed] = best_val_f1
    print(f"  seed={seed:5d}  val_f1={best_val_f1:.4f}  ✅ 儲存")

print(f"\n{'='*50}")
print(f"  所有 seeds 結果：")
for s, f1 in sorted(seed_results.items(), key=lambda x: -x[1]):
    print(f"    seed={s:5d}  val_f1={f1:.4f}")
best_seed = max(seed_results, key=seed_results.get)
print(f"\n  🏆 最佳單一 seed：{best_seed}（val_f1={seed_results[best_seed]:.4f}）")
print(f"\n下一步：5_evaluate_clf.py（ensemble + 三層評估）")
