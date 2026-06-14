"""
Step 4：BiLSTM Aligned Finetune（Multi-task + SMOTE + 固定 class_weights）
"""

import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_bilstm_v2"

# ── 超參數 ─────────────────────────────────────────
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
MT_ALPHA      = 0.5
FOCAL_GAMMA   = 2.0

SEEDS = [42, 123, 777]

# ── Device ─────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ── SMOTE ─────────────────────────────────────────
def apply_smote(X_train, y_train, labels):
    n, t, f = X_train.shape

    print("\n🔁 SMOTE 前:", Counter(labels.tolist()))

    X_flat = X_train.reshape(n, -1)
    Xy = np.concatenate([X_flat, y_train.reshape(-1, 1)], axis=1)

    min_count = min(Counter(labels.tolist()).values())
    if min_count <= 1:
        print("⚠️ 類別太少，跳過 SMOTE")
        return X_train, y_train, labels

    k = min(3, min_count - 1)

    sm = SMOTE(sampling_strategy="not majority", k_neighbors=k, random_state=42)
    Xy_res, labels_res = sm.fit_resample(Xy, labels)

    X_res = Xy_res[:, :-1].reshape(-1, t, f)
    y_res = Xy_res[:, -1].reshape(-1, 1)

    print("🔁 SMOTE 後:", Counter(labels_res.tolist()))

    return X_res.astype(np.float32), y_res.astype(np.float32), labels_res


# ── Loss ─────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Model ─────────────────────────────────────────
class BiLSTMWithAttentionMT(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )

        bi_hidden = HIDDEN_SIZE * 2

        self.attn = nn.Linear(bi_hidden, 1)
        self.norm = nn.LayerNorm(bi_hidden)
        self.drop = nn.Dropout(DROPOUT)

        self.fc1 = nn.Linear(bi_hidden, HIDDEN_SIZE)
        self.reg = nn.Linear(HIDDEN_SIZE, 1)
        self.cls = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

        self.relu = nn.ReLU()

    def encode(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        return self.norm((out * w).sum(dim=1))

    def forward(self, x):
        h = self.relu(self.fc1(self.drop(self.encode(x))))
        return self.reg(h), self.cls(h)


# ── Load pretrained ───────────────────────────────
def load_pretrained():
    ckpt = torch.load(f"{ARTIFACTS_DIR}/pretrain_bilstm.pth", map_location=device)

    model = BiLSTMWithAttentionMT()
    state = model.state_dict()

    for k, v in ckpt["model_state"].items():
        if k in state and state[k].shape == v.shape:
            state[k] = v

    model.load_state_dict(state)
    return model


# ── Load data ─────────────────────────────────────
print("📂 載入資料...")

X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")

train_labels = np.load(f"{ARTIFACTS_DIR}/personal_y_train_risk_labels.npy")
val_labels   = np.load(f"{ARTIFACTS_DIR}/personal_y_val_risk_labels.npy")

# ── SMOTE ─────────────────────────────────────────
X_train, y_train, train_labels = apply_smote(X_train, y_train, train_labels)

# ── ❗固定 class_weights（你指定的） ─────────────────
class_weights = torch.tensor([1.0, 3.0, 4.0, 1.0], dtype=torch.float32)

print("\n🎯 使用固定 class_weights:", class_weights.numpy())

# ── DataLoader ────────────────────────────────────
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(train_labels, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val),
        torch.tensor(val_labels, dtype=torch.long),
    ),
    batch_size=BATCH_SIZE
)

# ── Training ──────────────────────────────────────
for seed in SEEDS:
    print(f"\n🚀 Seed {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_pretrained().to(device)

    huber = nn.HuberLoss(delta=HUBER_DELTA)
    focal = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights.to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=7)

    best = float("inf")
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for Xb, yb, lb in train_loader:
            Xb, yb, lb = Xb.to(device), yb.to(device), lb.to(device)

            opt.zero_grad()

            reg, cls = model(Xb)
            loss = huber(reg, yb) + MT_ALPHA * focal(cls, lb)

            loss.backward()
            opt.step()

            total_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for Xb, yb, _ in val_loader:
                reg, _ = model(Xb.to(device))
                val_loss += huber(reg, yb.to(device)).item()

        val_loss /= len(val_loader)
        sch.step(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train {total_loss:.4f} | Val {val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            patience = 0
            torch.save(
                {"model_state": model.state_dict()},
                f"{ARTIFACTS_DIR}/finetune_bilstm_smote_seed{seed}.pth"
            )
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stop")
                break

print("\n🎉 完成 SMOTE + 固定權重訓練")