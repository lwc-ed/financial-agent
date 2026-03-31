"""版本 V4 預測評估腳本（LLRD + 混合損失）"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle, json, os
from datetime import datetime

ARTIFACTS_DIR = "artificats"
VERSION       = "v4"
RESULT_PATH   = "result_v4.txt"

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


print(f"📦 載入 V{VERSION} 模型...")
ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_gru_{VERSION}.pth", map_location=device)
hp    = ckpt["hyperparams"]
model = GRUWithAttention(hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

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

def predict_real(X):
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return target_scaler.inverse_transform(pred)

y_val_pred  = predict_real(X_val)
y_val_true  = target_scaler.inverse_transform(y_val)
y_test_pred = predict_real(X_test)
y_test_true = target_scaler.inverse_transform(y_test)

val_mae   = float(np.mean(np.abs(y_val_pred  - y_val_true)))
val_rmse  = float(np.sqrt(np.mean((y_val_pred  - y_val_true)**2)))
test_mae  = float(np.mean(np.abs(y_test_pred - y_test_true)))
test_rmse = float(np.sqrt(np.mean((y_test_pred - y_test_true)**2)))
val_smape  = smape(y_val_true,  y_val_pred)
test_smape = smape(y_test_true, y_test_pred)
val_nmae   = per_user_nmae(y_val_true,  y_val_pred,  val_user_ids)
test_nmae  = per_user_nmae(y_test_true, y_test_pred, test_user_ids)

EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]
df = pd.read_csv("features_all.csv")
df["date"] = pd.to_datetime(df["date"])
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
df["naive_7d"]  = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(7, min_periods=1).sum())
df["ma_30d_x7"] = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(30, min_periods=1).mean()) * 7
naive_mae  = float((df["naive_7d"]  - df["future_expense_7d_sum"]).abs().mean())
ma_mae     = float((df["ma_30d_x7"] - df["future_expense_7d_sum"]).abs().mean())
naive_rmse = float(((df["naive_7d"]  - df["future_expense_7d_sum"])**2).mean()**0.5)
ma_rmse    = float(((df["ma_30d_x7"] - df["future_expense_7d_sum"])**2).mean()**0.5)

print(f"\n{'='*50}  V{VERSION} Results  {'='*50}")
print(f"  Val  MAE : {val_mae:,.2f}  RMSE: {val_rmse:,.2f}  SMAPE: {val_smape:.2f}%  NMAE: {val_nmae:.2f}%")
print(f"  Test MAE : {test_mae:,.2f}  RMSE: {test_rmse:,.2f}  SMAPE: {test_smape:.2f}%  NMAE: {test_nmae:.2f}%")
print(f"  Beat baseline: {test_mae < ma_mae}")

for v in ["v1", "v2", "v3"]:
    path = f"{ARTIFACTS_DIR}/metrics_{v}.json"
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        print(f"  vs {v}: Test MAE {prev.get('test_mae',0):,.2f} → {test_mae:,.2f} | SMAPE {prev.get('test_smape',0):.2f}% → {test_smape:.2f}%")

metrics = {
    "model_name": f"gru_v{VERSION}", "version": VERSION,
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

test_pred_path = f"{ARTIFACTS_DIR}/predictions_test_{VERSION}.csv"
pd.DataFrame({"y_true": y_test_true.flatten(), "y_pred": y_test_pred.flatten(),
              "abs_error": np.abs(y_test_pred - y_test_true).flatten()}).to_csv(
    test_pred_path, index=False)

metrics_path = f"{ARTIFACTS_DIR}/metrics_{VERSION}.json"
result_text = f"""GRU Predict Result
model_name: gru_{VERSION}
version: {VERSION}
val_mae: {val_mae:,.6f}
val_rmse: {val_rmse:,.6f}
val_smape: {val_smape:.2f}%
val_per_user_nmae: {val_nmae:.2f}%
test_mae: {test_mae:,.6f}
test_rmse: {test_rmse:,.6f}
test_smape: {test_smape:.2f}%
test_per_user_nmae: {test_nmae:.2f}%
naive_7d_mae: {naive_mae:,.6f}
naive_7d_rmse: {naive_rmse:,.6f}
moving_avg_30d_mae: {ma_mae:,.6f}
moving_avg_30d_rmse: {ma_rmse:,.6f}
beat_moving_avg: {test_mae < ma_mae}
metrics_json: {metrics_path}
test_predictions_csv: {test_pred_path}
"""
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"\n✅ V{VERSION} metrics 已儲存：{metrics_path}")
print(f"📝 V{VERSION} Result 已儲存：{RESULT_PATH}")
