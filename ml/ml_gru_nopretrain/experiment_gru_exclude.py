"""
experiment_gru_exclude.py
==========================
比較 GRU from scratch 在不同用戶子集下的表現：

  情境 A：全 16 人（參考，直接讀上次結果）
  情境 B：排除 user4/5/6（13 人）
  情境 C：排除 user4/5/6 + user14（12 人）

同時附上 HGBR 22特徵的對應子集結果做對比。
"""

import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from datetime import datetime

os.makedirs("artifacts", exist_ok=True)

DATA_PATH            = "../ml_gru/features_all.csv"
INPUT_DAYS           = 30
USER_CLIP_PERCENTILE = 99
TARGET_COL           = "future_expense_7d_sum"

GRU_FEATURES = [
    "daily_expense", "expense_7d_mean", "expense_30d_sum",
    "has_expense", "has_income", "net_30d_sum", "txn_30d_sum",
]
HGBR_FEATURES = [
    "daily_expense", "daily_income", "daily_net",
    "has_expense", "has_income",
    "dow", "is_weekend", "day", "month",
    "is_summer_vacation", "is_winter_vacation", "days_to_end_of_month",
    "expense_7d_sum", "expense_7d_mean", "net_7d_sum", "txn_7d_sum",
    "expense_30d_sum", "expense_30d_mean", "net_30d_sum", "txn_30d_sum",
    "expense_7d_30d_ratio", "expense_trend",
]

# GRU 超參數
INPUT_SIZE   = 7
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
OUTPUT_SIZE  = 1
DROPOUT      = 0.4
BATCH_SIZE   = 16
EPOCHS       = 200
PATIENCE     = 30
LR           = 1e-3
WEIGHT_DECAY = 5e-4

if torch.backends.mps.is_available():   device = torch.device("mps")
elif torch.cuda.is_available():         device = torch.device("cuda")
else:                                   device = torch.device("cpu")

# ─────────────────────────────────────────────
# 模型定義
# ─────────────────────────────────────────────
class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)
        self.relu       = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        return self.fc2(self.relu(self.fc1(self.dropout(context))))

# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────
def smape(yt, yp):
    yt, yp = yt.flatten(), yp.flatten()
    d = (np.abs(yt) + np.abs(yp)) / 2
    m = d > 0
    return float(np.mean(np.abs(yp[m] - yt[m]) / d[m]) * 100)

def per_user_nmae(yt, yp, uids):
    r = []
    for u in np.unique(uids):
        mask = np.array(uids) == u
        mu = yt[mask].mean()
        if mu > 0:
            r.append(np.mean(np.abs(yp[mask] - yt[mask])) / mu * 100)
    return float(np.mean(r))

def build_dataset(df_src):
    """建立 GRU 序列 + HGBR 扁平資料"""
    gXtr, gXva, gXte = [], [], []
    hXtr, hXva, hXte = [], [], []
    ytr, yva, yte     = [], [], []
    tr_u, va_u, te_u  = [], [], []
    clip_map = {}

    for uid in df_src["user_id"].unique():
        u         = df_src[df_src["user_id"] == uid].reset_index(drop=True)
        gru_f     = u[GRU_FEATURES].values.astype(np.float32)
        hgbr_f    = u[HGBR_FEATURES].values.astype(np.float32)
        target    = u[TARGET_COL].values.astype(np.float32)

        gru_wins, hgbr_rows, y_vals = [], [], []
        for t in range(INPUT_DAYS, len(u)):
            gru_wins.append(gru_f[t - INPUT_DAYS:t])
            hgbr_rows.append(hgbr_f[t])
            y_vals.append(target[t])

        if len(gru_wins) == 0:
            continue
        n     = len(gru_wins)
        t_end = int(n * 0.70)
        v_end = int(n * 0.85)
        if t_end == 0:
            continue

        gw  = np.array(gru_wins,  dtype=np.float32)
        hf  = np.array(hgbr_rows, dtype=np.float32)
        ya  = np.array(y_vals,    dtype=np.float32)

        # Per-user P99 clipping on GRU features
        clip_vals = {col: float(np.percentile(gw[:t_end, :, i], USER_CLIP_PERCENTILE))
                     for i, col in enumerate(GRU_FEATURES)}
        clip_map[str(uid)] = clip_vals

        def clip(arr):
            out = arr.copy()
            for i, col in enumerate(GRU_FEATURES):
                out[:, :, i] = np.clip(out[:, :, i], None, clip_vals[col])
            return out

        gXtr.extend(clip(gw[:t_end]));      gXva.extend(clip(gw[t_end:v_end]));   gXte.extend(clip(gw[v_end:]))
        hXtr.extend(hf[:t_end]);            hXva.extend(hf[t_end:v_end]);          hXte.extend(hf[v_end:])
        ytr.extend(ya[:t_end]);             yva.extend(ya[t_end:v_end]);            yte.extend(ya[v_end:])
        tr_u.extend([uid]*t_end);           va_u.extend([uid]*(v_end-t_end));       te_u.extend([uid]*(n-v_end))

    gXtr = np.array(gXtr, dtype=np.float32); gXva = np.array(gXva, dtype=np.float32); gXte = np.array(gXte, dtype=np.float32)
    hXtr = np.array(hXtr, dtype=np.float32); hXva = np.array(hXva, dtype=np.float32); hXte = np.array(hXte, dtype=np.float32)
    ytr  = np.array(ytr,  dtype=np.float32).reshape(-1,1)
    yva  = np.array(yva,  dtype=np.float32).reshape(-1,1)
    yte  = np.array(yte,  dtype=np.float32).reshape(-1,1)

    # GRU scaler
    gru_sc = StandardScaler()
    gru_sc.fit(gXtr.reshape(-1, len(GRU_FEATURES)))
    gXtr = gru_sc.transform(gXtr.reshape(-1,len(GRU_FEATURES))).reshape(gXtr.shape).astype(np.float32)
    gXva = gru_sc.transform(gXva.reshape(-1,len(GRU_FEATURES))).reshape(gXva.shape).astype(np.float32)
    gXte = gru_sc.transform(gXte.reshape(-1,len(GRU_FEATURES))).reshape(gXte.shape).astype(np.float32)

    tgt_sc = StandardScaler()
    tgt_sc.fit(ytr)
    ytr_sc = tgt_sc.transform(ytr).astype(np.float32)
    yva_sc = tgt_sc.transform(yva).astype(np.float32)

    # HGBR scaler
    hgbr_sc = StandardScaler()
    hgbr_sc.fit(hXtr)
    hXtr = hgbr_sc.transform(hXtr).astype(np.float32)
    hXva = hgbr_sc.transform(hXva).astype(np.float32)
    hXte = hgbr_sc.transform(hXte).astype(np.float32)

    return {
        "gXtr": gXtr, "gXva": gXva, "gXte": gXte,
        "hXtr": hXtr, "hXva": hXva, "hXte": hXte,
        "ytr": ytr, "yva": yva, "yte": yte,
        "ytr_sc": ytr_sc, "yva_sc": yva_sc,
        "tr_u": np.array(tr_u), "va_u": np.array(va_u), "te_u": np.array(te_u),
        "gru_sc": gru_sc, "tgt_sc": tgt_sc, "hgbr_sc": hgbr_sc,
    }

def train_gru(d, label):
    """訓練 GRU，回傳 test 預測（原始空間）"""
    loader_tr = DataLoader(TensorDataset(torch.tensor(d["gXtr"]), torch.tensor(d["ytr_sc"])),
                           batch_size=BATCH_SIZE, shuffle=True)
    loader_va = DataLoader(TensorDataset(torch.tensor(d["gXva"]), torch.tensor(d["yva_sc"])),
                           batch_size=BATCH_SIZE, shuffle=False)

    model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    huber = nn.HuberLoss(delta=1.0)
    mse   = nn.MSELoss()
    crit  = lambda p,t: 0.7*huber(p,t) + 0.3*mse(p,t)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=2, eta_min=1e-7)

    best_loss, best_state, no_improve = float("inf"), None, 0
    print(f"  訓練中（{len(d['gXtr'])} 筆 train）...", end="", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in loader_tr:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in loader_va:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += crit(model(Xb), yb).item()
        val_loss /= len(loader_va)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f" early stop @ ep{epoch}", end="")
                break

    print(f" 完成（best val_loss={best_loss:.4f}）")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_sc = model(torch.tensor(d["gXte"], dtype=torch.float32).to(device)).cpu().numpy()
    return d["tgt_sc"].inverse_transform(pred_sc).flatten()

def train_hgbr(d):
    Xtv = np.concatenate([d["hXtr"], d["hXva"]])
    ytv = np.concatenate([d["ytr"].flatten(), d["yva"].flatten()])
    m = HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=42, verbose=0)
    m.fit(Xtv, ytv)
    return m.predict(d["hXte"])

def evaluate(y_true, y_pred, uids):
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    return {"mae": round(mae,2), "rmse": round(rmse,2),
            "smape": round(smape(y_true, y_pred),4),
            "nmae":  round(per_user_nmae(y_true, y_pred, uids),4)}

# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
print("📂 讀取資料...")
df_all = pd.read_csv(DATA_PATH)
df_all["date"] = pd.to_datetime(df_all["date"])
df_all = df_all.sort_values(["user_id","date"]).reset_index(drop=True)

SCENARIOS = {
    "B_excl_456":     ["user4","user5","user6"],
    "C_excl_456_14":  ["user4","user5","user6","user14"],
}

all_results = {}

# 情境 A：全 16 人（直接載入上次結果）
prev = json.load(open("artifacts/experiment_exclude_users.json"))
print("\n情境 A 的 HGBR 結果直接從上次實驗載入")
all_results["A_all16"] = {
    "users": 16, "exclude": [],
    "hgbr": prev["scenario_A_16users_train_13users_eval"],   # 評估13人子集
}
# GRU 16人的結果從之前的 comparison_results 取
cr = json.load(open("artifacts/comparison_results.json"))
all_results["A_all16"]["gru"] = cr["models"]["GRU from scratch"]

for label, excl in SCENARIOS.items():
    n_keep = 16 - len(excl)
    df_sub = df_all[~df_all["user_id"].isin(excl)].copy()
    print(f"\n{'='*60}")
    print(f"情境 {label}：排除 {excl}，保留 {n_keep} 人")
    print(f"{'='*60}")

    d = build_dataset(df_sub)
    yte_raw = d["yte"].flatten()
    te_uids = d["te_u"]

    print(f"  [HGBR 22特徵]", end="", flush=True)
    hgbr_pred = train_hgbr(d)
    print(f"  完成")

    print(f"  [GRU from scratch]")
    gru_pred  = train_gru(d, label)

    hgbr_res = evaluate(yte_raw, hgbr_pred, te_uids)
    gru_res  = evaluate(yte_raw, gru_pred,  te_uids)

    # Per-user 明細
    peruser = {}
    for uid in sorted(set(te_uids), key=str):
        m = te_uids == uid
        peruser[str(uid)] = {
            "n": int(m.sum()),
            "hgbr_mae": round(float(np.mean(np.abs(hgbr_pred[m] - yte_raw[m]))), 2),
            "gru_mae":  round(float(np.mean(np.abs(gru_pred[m]  - yte_raw[m]))), 2),
        }

    all_results[label] = {
        "users": n_keep, "exclude": excl,
        "hgbr": hgbr_res, "gru": gru_res,
        "per_user": peruser,
    }

    print(f"\n  HGBR  Test MAE: {hgbr_res['mae']:,.0f}  RMSE: {hgbr_res['rmse']:,.0f}  SMAPE: {hgbr_res['smape']:.2f}%")
    print(f"  GRU   Test MAE: {gru_res['mae']:,.0f}  RMSE: {gru_res['rmse']:,.0f}  SMAPE: {gru_res['smape']:.2f}%")

# ─────────────────────────────────────────────
# 綜合比較表
# ─────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  綜合比較（評估對象 = 各情境保留的使用者）")
print(f"  {'情境':<22} {'用戶數':>6} {'HGBR MAE':>10} {'GRU MAE':>10} {'HGBR 贏?':>10}")
print(f"  {'─'*60}")

# 情境A 只看13人子集的 GRU，需重算
# 從上次 predictions_test.csv 取 GRU 13人結果
try:
    pred_df = pd.read_csv("artifacts/predictions_test.csv")
    pred_13 = pred_df[~pred_df["user_id"].isin(["user4","user5","user6"])]
    gru_A_13_mae = float(np.mean(np.abs(pred_13["gru_scratch_pred"] - pred_13["y_true"])))
    hgbr_A_13_mae = prev["scenario_A_16users_train_13users_eval"]["test_mae"]
except:
    gru_A_13_mae   = cr["models"]["GRU from scratch"]["test_mae"]
    hgbr_A_13_mae  = prev["scenario_A_16users_train_13users_eval"]["test_mae"]

rows = [
    ("A 全16人訓練（13人eval）", 16, hgbr_A_13_mae, gru_A_13_mae),
]
for label, excl in SCENARIOS.items():
    r = all_results[label]
    rows.append((label, r["users"], r["hgbr"]["mae"], r["gru"]["mae"]))

for name, nu, hm, gm in rows:
    flag = "✅" if hm < gm else "❌"
    print(f"  {name:<22} {nu:>6} {hm:>10,.0f} {gm:>10,.0f} {flag:>10}")

# Per-user 明細（情境B和C）
for label in SCENARIOS:
    r = all_results[label]
    print(f"\n  Per-user（{label}）：")
    print(f"  {'User':<10} {'n':>5} {'HGBR':>9} {'GRU':>9} {'HGBR贏':>7}")
    for uid, v in sorted(r["per_user"].items()):
        flag = "✅" if v["hgbr_mae"] < v["gru_mae"] else "❌"
        print(f"  {uid:<10} {v['n']:>5} {v['hgbr_mae']:>9,.0f} {v['gru_mae']:>9,.0f} {flag:>7}")

# 儲存
out = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "scenarios": all_results,
    "summary": [{"scenario": n, "n_users": nu, "hgbr_mae": hm, "gru_mae": gm}
                for n, nu, hm, gm in rows],
}
with open("artifacts/experiment_gru_exclude.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n✅ 結果已儲存至 artifacts/experiment_gru_exclude.json")
