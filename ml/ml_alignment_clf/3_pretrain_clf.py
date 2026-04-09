"""
Step 3：Walmart 預訓練（分類任務）
=====================================
以 Walmart 資料預訓練「消費異常預警分類器」
Loss：BCEWithLogitsLoss（含 pos_weight 處理類別不平衡）

Transfer Learning 故事：
  Walmart 有大量歷史資料，先讓模型學會「什麼樣的消費模式會在接下來 7 天超標」
  再將學到的 encoder 知識遷移到個人資料（資料量少但任務相同）

輸出：artifacts_clf/pretrain_clf_gru.pth
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_clf"

INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)   # 12
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
BATCH_SIZE    = 64
EPOCHS        = 150
LR            = 0.0001
PATIENCE      = 20
WEIGHT_DECAY  = 5e-4

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ 使用 CUDA")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")


class GRUClassifier(nn.Module):
    """
    GRU + Temporal Attention → Binary Classifier
    輸出：logit（使用 BCEWithLogitsLoss，不需要 sigmoid）
    Encoder（gru + attention + layer_norm）與 regression 版架構一致，
    方便未來做 ablation 比較。
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, 1)   # logit
        self.relu       = nn.ReLU()

    def encode(self, x):
        """只回傳 encoder 的 context vector（供 MMD 計算）"""
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        return context

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)   # shape: (B, 1)


# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("\n📂 載入 Walmart 分類資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/walmart_clf_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/walmart_clf_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/walmart_clf_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/walmart_clf_y_val.npy")
pos_weight_val = float(np.load(f"{ARTIFACTS_DIR}/walmart_clf_pos_weight.npy")[0])
print(f"  X_train: {X_train.shape}  pos_weight={pos_weight_val:.3f}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=False
)

model     = GRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
pos_w_t   = torch.tensor([pos_weight_val]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  GRUClassifier 參數量：{total_params:,}")
print(f"   input={INPUT_SIZE}(特徵)  hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}")

# ── 訓練 ──────────────────────────────────────────────────────────────────────
print(f"\n🚀 開始 Pretrain（Walmart，BCE Loss，pos_weight={pos_weight_val:.3f}）...")
print("=" * 65)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/pretrain_clf_gru.pth"
train_losses, val_losses = [], []

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

    # 計算 val F1
    vp, vl = np.array(val_preds), np.array(val_labels)
    tp = int(np.sum((vp == 1) & (vl == 1)))
    fp = int(np.sum((vp == 1) & (vl == 0)))
    fn = int(np.sum((vp == 0) & (vl == 1)))
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    lr = optimizer.param_groups[0]["lr"]

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.5f} | Val: {val_loss:.5f} "
              f"| Val F1: {f1:.4f} | LR: {lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "val_loss"    : best_val_loss,
            "val_f1"      : f1,
            "hyperparams" : {
                "input_size"   : INPUT_SIZE,
                "hidden_size"  : HIDDEN_SIZE,
                "num_layers"   : NUM_LAYERS,
                "dropout"      : DROPOUT,
                "feature_cols" : ALIGNED_FEATURE_COLS,
                "task"         : "classification",
            }
        }, best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping（patience={PATIENCE}）在 epoch {epoch}")
            break

print("=" * 65)
print(f"\n🎉 Pretrain 完成！最佳 Val Loss：{best_val_loss:.5f}")
print(f"  ✅ {best_model_path}")
print(f"\n下一步：4_finetune_clf.py")
