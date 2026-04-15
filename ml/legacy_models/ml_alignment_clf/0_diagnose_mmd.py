"""
Step 0：MMD 診斷（分類任務版）
===============================
計算 Walmart 與個人資料在 encoder 表示空間中的 MMD（Maximum Mean Discrepancy）
使用：已完成 fine-tune 的模型 encoder（而非 raw features）

為什麼算 encoder 表示的 MMD 比 raw features 更有說服力？
  - raw features 的 MMD 顯示「輸入分佈的對齊程度」
  - encoder 表示的 MMD 顯示「模型學到的表示是否跨域一致」
  - 後者更直接說明 Transfer Learning 的有效性

輸出：artifacts_clf/mmd_diagnosis.json
"""

import numpy as np
import torch
import torch.nn as nn
import glob, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import (
    ALIGNED_FEATURE_COLS, load_walmart_daily, load_personal_daily,
    compute_aligned_features, INPUT_DAYS
)

ARTIFACTS_DIR = "artifacts_clf"
INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
N_SAMPLE      = 500   # 每個 domain 取多少 sample 計算 MMD

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """RBF kernel MMD（使用 numpy，避免大型矩陣 OOM）"""
    def rbf_kernel(A, B, g):
        diff = A[:, None, :] - B[None, :, :]
        dist2 = (diff ** 2).sum(axis=2)
        return np.exp(-g * dist2)
    Kxx = rbf_kernel(X, X, gamma).mean()
    Kyy = rbf_kernel(Y, Y, gamma).mean()
    Kxy = rbf_kernel(X, Y, gamma).mean()
    return float(Kxx + Kyy - 2 * Kxy)


def extract_features_sliding(series, dates):
    """從時間序列萃取 aligned features 並做滑動視窗"""
    feats = compute_aligned_features(series, dates)
    arr   = feats[ALIGNED_FEATURE_COLS].values.astype(np.float32)
    windows = []
    for t in range(INPUT_DAYS, len(arr)):
        windows.append(arr[t - INPUT_DAYS : t])
    return np.array(windows, dtype=np.float32) if windows else None


# ── 從 raw features 估計 MMD（不需要模型）─────────────────────────────────────
print("📂 載入 Walmart & 個人資料（計算 raw feature MMD）...")
walmart_df  = load_walmart_daily()
personal_df = load_personal_daily()

walmart_X_list, personal_X_list = [], []

for store_id in list(walmart_df["store_id"].unique())[:15]:   # 取 15 家門市
    u = walmart_df[walmart_df["store_id"] == store_id].sort_values("date").reset_index(drop=True)
    windows = extract_features_sliding(u["daily_expense"], u["date"])
    if windows is not None and len(windows) > 0:
        walmart_X_list.append(windows)

for user_id in personal_df["user_id"].unique():
    u = personal_df[personal_df["user_id"] == user_id].sort_values("date").reset_index(drop=True)
    windows = extract_features_sliding(u["daily_expense"], u["date"])
    if windows is not None and len(windows) > 0:
        personal_X_list.append(windows)

walmart_X  = np.concatenate(walmart_X_list, axis=0)   # (N_w, T, F)
personal_X = np.concatenate(personal_X_list, axis=0)   # (N_p, T, F)

# 取最後一天的特徵向量做 MMD（代表當前狀態分佈）
W_raw = walmart_X[:, -1, :]   # (N_w, F)
P_raw = personal_X[:, -1, :]  # (N_p, F)

np.random.seed(42)
idx_w = np.random.choice(len(W_raw), min(N_SAMPLE, len(W_raw)), replace=False)
idx_p = np.random.choice(len(P_raw), min(N_SAMPLE, len(P_raw)), replace=False)
mmd_raw = mmd_rbf(W_raw[idx_w], P_raw[idx_p], gamma=1.0)
print(f"  Raw feature MMD（最後一天特徵向量）: {mmd_raw:.6f}")

# ── 從模型 encoder 表示計算 MMD（需要 fine-tuned 模型）──────────────────────
SEEDS = sorted([
    int(f.split("seed")[1].replace(".pth", ""))
    for f in glob.glob(f"{ARTIFACTS_DIR}/finetune_clf_seed*.pth")
])

mmd_encoded_list = []
if SEEDS:
    print(f"\n🔮 計算 Encoder 表示 MMD（{len(SEEDS)} 個 seeds 平均）...")
    for seed in SEEDS:
        ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_clf_seed{seed}.pth", map_location=device)
        model = GRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            W_t = torch.tensor(walmart_X[idx_w], dtype=torch.float32).to(device)
            P_t = torch.tensor(personal_X[idx_p], dtype=torch.float32).to(device)
            W_enc = model.encode(W_t).cpu().numpy()
            P_enc = model.encode(P_t).cpu().numpy()

        mmd_enc = mmd_rbf(W_enc, P_enc, gamma=1.0 / HIDDEN_SIZE)
        mmd_encoded_list.append(mmd_enc)
        print(f"  seed={seed}  Encoder MMD={mmd_enc:.6f}")

    mmd_encoded_mean = float(np.mean(mmd_encoded_list))
    print(f"\n  Encoder MMD 均值（{len(SEEDS)} seeds）: {mmd_encoded_mean:.6f}")
    print(f"  （MMD 越低 → 兩個 domain 的 encoder 表示越對齊 → TL 效果越好）")
else:
    mmd_encoded_mean = None
    print("  ⚠️  尚無 fine-tuned 模型，只計算 raw feature MMD")

# ── 輸出 ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  MMD 診斷結果")
print(f"{'='*55}")
print(f"  Raw feature MMD (gamma=1.0)  : {mmd_raw:.6f}")
if mmd_encoded_mean is not None:
    print(f"  Encoder repr MMD (gamma=1/H) : {mmd_encoded_mean:.6f}")
print(f"  特徵數量                     : {len(ALIGNED_FEATURE_COLS)} (含 spike 特徵)")
print(f"{'='*55}")
print(f"  解讀：值越接近 0 → 兩 domain 分佈越相似")
print(f"        建議範圍 < 0.05 表示良好對齊")

output = {
    "n_sample_per_domain" : N_SAMPLE,
    "n_features"          : len(ALIGNED_FEATURE_COLS),
    "feature_cols"        : ALIGNED_FEATURE_COLS,
    "mmd_raw_features"    : round(mmd_raw, 6),
    "mmd_encoded_repr"    : round(mmd_encoded_mean, 6) if mmd_encoded_mean is not None else None,
    "seeds_used"          : SEEDS,
    "interpretation"      : "MMD < 0.05 表示良好的域對齊（domain alignment），支持 Transfer Learning 的有效性",
}
with open(f"{ARTIFACTS_DIR}/mmd_diagnosis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 儲存至 {ARTIFACTS_DIR}/mmd_diagnosis.json")
print("🎉 完成！")
