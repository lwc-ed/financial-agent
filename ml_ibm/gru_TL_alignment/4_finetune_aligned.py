"""
Step 4：GRU Finetune（IBM TL，Multi-task）
==========================================
載入 IBM Pretrained GRU，在個人資料上做 finetune
Loss = HuberLoss + MT_ALPHA × FocalLoss（4-class risk level）
  - 回歸頭：預測未來 7 天花費金額
  - 分類頭：同時預測 risk level（no_alarm / low / mid / high）
  - class-weighted Focal Loss（gamma=2）處理少數 class
Ensemble 多 seeds
輸出：
  - finetune_aligned_gru_seed{seed}.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_aligned"

INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
NUM_CLASSES   = 4
BATCH_SIZE    = 32
EPOCHS        = 80
LEARNING_RATE = 1e-4
PATIENCE      = 20
WEIGHT_DECAY  = 1e-4
HUBER_DELTA   = 1.0
MT_ALPHA      = 0.5   # classification loss 的權重
FOCAL_GAMMA   = 2.0   # focal loss gamma
SEEDS = [
    42, 123, 777, 456, 789, 999, 2024,
    0, 7, 13, 21, 100, 314, 1234, 9999,
    11, 22, 33, 44, 55, 66, 77, 88, 99,
    111, 222, 333, 444, 555, 666
]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
else:
    device = torch.device("cpu")

print("📂 載入個人資料...")
X_train      = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_train.npy")
y_train      = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_train.npy")
X_val        = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_val.npy")
y_val        = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_val.npy")
train_labels = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_train_risk_labels.npy")
val_labels   = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_val_risk_labels.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")
print(f"  Train risk 分佈: {dict(sorted(Counter(train_labels.tolist()).items()))}")

# ── Class weights ─────────────────────────────────────────────────────────────
label_counts  = Counter(train_labels.tolist())
total_samples = len(train_labels)
class_weights = torch.tensor(
    [total_samples / (NUM_CLASSES * label_counts.get(i, 1)) for i in range(NUM_CLASSES)],
    dtype=torch.float32
)
print(f"  Focal class weights: {class_weights.numpy().round(2)}")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ── 模型架構（Multi-task）─────────────────────────────────────────────────────
class GRUWithAttentionMT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, num_classes=4):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)   # 回歸頭
        self.cls_head   = nn.Linear(hidden_size // 2, num_classes)   # 分類頭
        self.relu       = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        hidden  = self.relu(self.fc1(out))
        return self.fc2(hidden), self.cls_head(hidden)


def load_pretrained_mt():
    """載入 pretrain 權重，新增分類頭（隨機初始化）"""
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/pretrain_aligned_gru.pth", map_location=device)
    model = GRUWithAttentionMT(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT, NUM_CLASSES)
    pretrained = ckpt["model_state"]
    current    = model.state_dict()
    for k, v in pretrained.items():
        if k in current:
            current[k] = v
    model.load_state_dict(current)
    return model


# ── Ensemble Finetune ─────────────────────────────────────────────────────────
print(f"\n🚀 Ensemble MT Finetune（seeds={SEEDS}，MT_ALPHA={MT_ALPHA}）...")

for seed in SEEDS:
    save_path = f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed{seed}.pth"

    if os.path.exists(save_path):
        print(f"\n  Seed {seed}：已存在，跳過")
        continue

    print(f"\n{'='*55}")
    print(f"  Seed {seed}")
    print(f"{'='*55}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model      = load_pretrained_mt().to(device)
    huber_crit = nn.HuberLoss(delta=HUBER_DELTA)
    ce_crit    = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights.to(device))
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
            torch.tensor(train_labels, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_val),
            torch.tensor(val_labels, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_huber = epoch_cls = 0.0

        for X_b, y_b, lbl_b in train_loader:
            X_b, y_b, lbl_b = X_b.to(device), y_b.to(device), lbl_b.to(device)
            optimizer.zero_grad()
            reg_out, cls_out = model(X_b)
            h_loss = huber_crit(reg_out, y_b)
            c_loss = ce_crit(cls_out, lbl_b)
            loss   = h_loss + MT_ALPHA * c_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_huber += h_loss.item()
            epoch_cls   += c_loss.item()

        epoch_huber /= len(train_loader)
        epoch_cls   /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b, lbl_b in val_loader:
                reg_out, _ = model(X_b.to(device))
                val_loss  += huber_crit(reg_out, y_b.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}  Huber: {epoch_huber:.4f}  CE: {epoch_cls:.4f}  Val: {val_loss:.6f}  LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : best_val_loss,
                "seed"       : seed,
                "version"    : "ibm_finetune_mt",
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹️  Early stopping")
                break

    print(f"  Seed {seed} 最佳 Val Loss: {best_val_loss:.6f}")

print(f"\n🎉 MT Finetune 完成！→ 下一步：執行 5_predict_aligned.py")
