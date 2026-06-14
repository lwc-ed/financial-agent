"""
Step 5：BiLSTM SMOTE Predict + Evaluation
=========================================
讀取 SMOTE finetune 模型 → test 預測 → ensemble → inverse scale → 共用 evaluator
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.output_eval_utils import run_output_evaluation

sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_bilstm_v2"

INPUT_SIZE  = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.4
OUTPUT_SIZE = 1
NUM_CLASSES = 4
BATCH_SIZE  = 128

SEEDS = [42, 123, 777]

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple M1 MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ CUDA")
else:
    device = torch.device("cpu")
    print("⚠️  CPU")


class BiLSTMWithAttentionMT(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            INPUT_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )

        bi_hidden = HIDDEN_SIZE * 2

        self.attn = nn.Linear(bi_hidden, 1)
        self.norm = nn.LayerNorm(bi_hidden)
        self.drop = nn.Dropout(DROPOUT)

        self.fc1 = nn.Linear(bi_hidden, HIDDEN_SIZE)
        self.reg = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.cls = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

        self.relu = nn.ReLU()

    def encode(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        context = (out * w).sum(dim=1)
        return self.norm(context)

    def forward(self, x):
        h = self.relu(self.fc1(self.drop(self.encode(x))))
        return self.reg(h), self.cls(h)


print("📂 載入 test data...")

X_test = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test_scaled = np.load(f"{ARTIFACTS_DIR}/personal_y_test.npy")
y_test_raw = np.load(f"{ARTIFACTS_DIR}/personal_y_test_raw.npy")

test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy", allow_pickle=True)
test_dates = np.load(f"{ARTIFACTS_DIR}/personal_test_dates.npy", allow_pickle=True)

train_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_train_user_ids.npy", allow_pickle=True)
train_dates = np.load(f"{ARTIFACTS_DIR}/personal_train_dates.npy", allow_pickle=True)

val_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy", allow_pickle=True)
val_dates = np.load(f"{ARTIFACTS_DIR}/personal_val_dates.npy", allow_pickle=True)

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

print(f"  X_test: {X_test.shape}")
print(f"  y_test_raw: {y_test_raw.shape}")

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
    batch_size=BATCH_SIZE,
    shuffle=False
)

all_preds_scaled = []

for seed in SEEDS:
    model_path = f"{ARTIFACTS_DIR}/finetune_bilstm_smote_seed{seed}.pth"

    if not os.path.exists(model_path):
        print(f"⚠️ 找不到模型，跳過 seed {seed}: {model_path}")
        continue

    print(f"\n🔮 載入 SMOTE 模型 seed {seed}")

    model = BiLSTMWithAttentionMT().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds = []

    with torch.no_grad():
        for (X_b,) in test_loader:
            X_b = X_b.to(device)
            reg_out, _ = model(X_b)
            preds.append(reg_out.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    all_preds_scaled.append(preds)

if len(all_preds_scaled) == 0:
    raise RuntimeError("沒有成功載入任何 SMOTE finetune 模型，請確認模型檔案是否存在。")

print("\n📊 Ensemble 平均預測...")
y_pred_scaled = np.mean(all_preds_scaled, axis=0)

y_pred_raw = target_scaler.inverse_transform(y_pred_scaled).reshape(-1)
y_true_raw = y_test_raw.reshape(-1)

y_pred_raw = np.maximum(y_pred_raw, 0)

prediction_input_df = pd.DataFrame({
    "user_id": test_user_ids,
    "date": pd.to_datetime(test_dates),
    "y_true": y_true_raw,
    "y_pred": y_pred_raw,
})

split_metadata_df = pd.concat([
    pd.DataFrame({
        "user_id": train_user_ids,
        "date": pd.to_datetime(train_dates),
        "split": "train",
    }),
    pd.DataFrame({
        "user_id": val_user_ids,
        "date": pd.to_datetime(val_dates),
        "split": "valid",
    }),
    pd.DataFrame({
        "user_id": test_user_ids,
        "date": pd.to_datetime(test_dates),
        "split": "test",
    }),
], ignore_index=True)

print("\n📈 執行共用 evaluator...")

run_output_evaluation(
    model_name="bilstm_TL_alignment_smote",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_root=Path(__file__).resolve().parents[1] / "model_outputs",
)

print("\n🎉 SMOTE prediction + evaluation 完成！")
print("輸出位置：ml_ibm/model_outputs/bilstm_TL_alignment_smote/")