"""版本 V3 預測評估腳本（Log1p 反轉換）"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle, json, os
from datetime import datetime

ARTIFACTS_DIR = "artificats"
VERSION       = "v3"

if torch.backends.mps.is_available():  device = torch.device("mps")
elif torch.cuda.is_available():        device = torch.device("cuda")
else:                                  device = torch.device("cpu")


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


print(f"📦 載入 V{VERSION} 模型（Log1p 空間）...")
ckpt = torch.load(f"{ARTIFACTS_DIR}/finetune_gru_{VERSION}.pth", map_location=device)
hp   = ckpt["hyperparams"]
model = GRUWithAttention(hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# V3 使用 log1p scaler（訓練在 log 空間）
with open(f"{ARTIFACTS_DIR}/log_target_scaler_v3.pkl", "rb") as f:
    log_scaler = pickle.load(f)
# 原始 scaler（取得 y_true 實際值）
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    orig_scaler = pickle.load(f)

X_val        = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val        = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_test       = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test       = np.load(f"{ARTIFACTS_DIR}/personal_y_test.npy")
X_train      = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train      = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
val_user_ids  = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",  allow_pickle=True)
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy", allow_pickle=True)

def smape(yt, yp):
    yt, yp = yt.flatten(), yp.flatten()
    d = (np.abs(yt) + np.abs(yp)) / 2
    m = d > 0
    return float(np.mean(np.abs(yp[m] - yt[m]) / d[m]) * 100)

def per_user_nmae(yt, yp, uids):
    yt, yp = yt.flatten(), yp.flatten()
    r = []
    for u in np.unique(uids):
        m = np.array(uids) == u
        mu = yt[m].mean()
        if mu > 0:
            r.append(np.mean(np.abs(yp[m] - yt[m])) / mu * 100)
    return float(np.mean(r))


def predict_real_logspace(X):
    """
    模型在 log1p 標準化空間輸出
    反轉換步驟：標準化 → log1p 空間 → expm1 → 實際金額
    """
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    pred_log = log_scaler.inverse_transform(pred_scaled)  # log1p 空間
    pred_log = np.clip(pred_log, 0, None)                 # 確保 log ≥ 0
    return np.expm1(pred_log)                             # 回到實際金額


def get_true_real(y_scaled):
    """y 是原始標準化，還原到實際金額"""
    real = orig_scaler.inverse_transform(y_scaled)
    return np.clip(real, 0, None)


y_val_pred  = predict_real_logspace(X_val)
y_val_true  = get_true_real(y_val)
y_test_pred = predict_real_logspace(X_test)
y_test_true = get_true_real(y_test)

val_mae   = float(np.mean(np.abs(y_val_pred  - y_val_true)))
val_rmse  = float(np.sqrt(np.mean((y_val_pred  - y_val_true)**2)))
test_mae  = float(np.mean(np.abs(y_test_pred - y_test_true)))
test_rmse = float(np.sqrt(np.mean((y_test_pred - y_test_true)**2)))
val_smape  = smape(y_val_true,  y_val_pred)
test_smape = smape(y_test_true, y_test_pred)
val_nmae   = per_user_nmae(y_val_true,  y_val_pred,  val_user_ids)
test_nmae  = per_user_nmae(y_test_true, y_test_pred, test_user_ids)

df = pd.read_csv("features_all.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
df["naive_7d"]  = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(7, min_periods=1).sum())
df["ma_30d_x7"] = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(30, min_periods=1).mean()) * 7
naive_mae  = float((df["naive_7d"]  - df["future_expense_7d_sum"]).abs().mean())
ma_mae     = float((df["ma_30d_x7"] - df["future_expense_7d_sum"]).abs().mean())
naive_rmse = float(((df["naive_7d"]  - df["future_expense_7d_sum"])**2).mean()**0.5)
ma_rmse    = float(((df["ma_30d_x7"] - df["future_expense_7d_sum"])**2).mean()**0.5)

print(f"\n{'='*50}  V{VERSION} Results (Log1p)  {'='*50}")
print(f"  Val  MAE : {val_mae:,.2f}  RMSE: {val_rmse:,.2f}  SMAPE: {val_smape:.2f}%  NMAE: {val_nmae:.2f}%")
print(f"  Test MAE : {test_mae:,.2f}  RMSE: {test_rmse:,.2f}  SMAPE: {test_smape:.2f}%  NMAE: {test_nmae:.2f}%")
print(f"  Baseline Moving Avg MAE: {ma_mae:,.2f}, Beat: {test_mae < ma_mae}")

for v in ["v1", "v2"]:
    path = f"{ARTIFACTS_DIR}/metrics_{v}.json"
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        print(f"  vs {v}: Test MAE {prev.get('test_mae',0):,.2f} → {test_mae:,.2f} | SMAPE {prev.get('test_smape',0):.2f}% → {test_smape:.2f}%")

metrics = {
    "model_name": f"gru_v{VERSION}_log1p", "version": VERSION,
    "val_mae": round(val_mae, 6), "val_rmse": round(val_rmse, 6),
    "val_smape": round(val_smape, 4), "val_per_user_nmae": round(val_nmae, 4),
    "test_mae": round(test_mae, 6), "test_rmse": round(test_rmse, 6),
    "test_smape": round(test_smape, 4), "test_per_user_nmae": round(test_nmae, 4),
    "naive_7d_mae": round(naive_mae, 6), "naive_7d_rmse": round(naive_rmse, 6),
    "moving_avg_30d_mae": round(ma_mae, 6), "moving_avg_30d_rmse": round(ma_rmse, 6),
    "beat_moving_avg": bool(test_mae < ma_mae),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(f"{ARTIFACTS_DIR}/metrics_v{VERSION}.json", "w") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame({"y_true": y_test_true.flatten(), "y_pred": y_test_pred.flatten(),
              "abs_error": np.abs(y_test_pred - y_test_true).flatten()}).to_csv(
    f"{ARTIFACTS_DIR}/predictions_test_v{VERSION}.csv", index=False)

print(f"\n✅ V{VERSION} metrics 已儲存")
