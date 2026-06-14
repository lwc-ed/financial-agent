import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ── 1. 路徑鎖定 ──────────────────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"
OUTPUT_BASE = MY_DIR.parent / "model_outputs" / "bigru_TL_alignment"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention

# ── 2. 超參數 ────────────────────────────────────────────────────────────
INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def amount_to_label(y_scaled):
    q1, q2, q3 = np.percentile(y_scaled, [50, 75, 90])
    labels = np.zeros_like(y_scaled)
    labels[y_scaled > q1] = 1; labels[y_scaled > q2] = 2; labels[y_scaled > q3] = 3
    return labels.flatten().astype(np.int64)

# ── 3. 載入資料 ──────────────────────────────────────────────────────────
X_test = np.load(ARTIFACTS_DIR / "personal_X_test.npy")
y_test_raw = np.load(ARTIFACTS_DIR / "personal_y_test.npy")
y_test_label = amount_to_label(y_test_raw)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test_label)), batch_size=BATCH_SIZE)

# ── 4. 推論 ──────────────────────────────────────────────────────────────
seeds = [42, 123, 777, 456, 789, 999, 2024]
all_probs = []

print(f"🔮 正在評估分類性能...")
for seed in seeds:
    weight_path = ARTIFACTS_DIR / f"finetune_bigru_cls_seed{seed}.pth"
    if not weight_path.exists(): continue
    
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size=1, dropout=DROPOUT)
    model.fc2 = nn.Linear(HIDDEN_SIZE, 4)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True)["model_state"])
    model.to(device).eval()
    
    seed_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            probs = torch.softmax(model(xb.to(device)), dim=1)
            seed_probs.append(probs.cpu().numpy())
    all_probs.append(np.concatenate(seed_probs, axis=0))

final_preds = np.argmax(np.mean(all_probs, axis=0), axis=1)

# ── 5. 輸出報告 ──────────────────────────────────────────────────────────
macro_f1 = f1_score(y_test_label, final_preds, average='macro')
print("\n📊 Classification Report:\n", classification_report(y_test_label, final_preds, target_names=['No','Low','Mid','High']))
print(f"🚀 Macro F1 Score: {macro_f1:.4f}")

metrics = {
    "macro_f1": float(macro_f1),
    "confusion_matrix": confusion_matrix(y_test_label, final_preds).tolist()
}
with open(OUTPUT_BASE / "metrics_classification.json", "w") as f:
    json.dump(metrics, f, indent=4)
print(f"✅ 報告已存至 metrics_classification.json")