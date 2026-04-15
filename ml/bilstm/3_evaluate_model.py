import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── 1. 重新定義相同的模型架構 ──────────────────────────────────────────
# PyTorch 需要知道空殼長什麼樣子，才能把我們存好的權重 (pth) 塞進去
class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(MyBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :] 
        out = self.fc1(last_time_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def main():
    print("🚀 [Step 3] 開始進行模型期末考 (最終評估)...")

    # ── 2. 路徑設定與載入測試資料 ────────────────────────────────────────
    # 直接定位到這支程式旁邊的 artifacts
    ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
    
    # 載入「沒看過」的測試資料，以及最重要的「真實金額」與「還原器」
    X_test = np.load(ARTIFACTS_DIR / "my_X_test.npy")
    y_test_raw = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy") 
    
    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    # ── 3. 載入訓練好的模型權重 ──────────────────────────────────────────
    INPUT_SIZE = X_test.shape[2]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyBiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    
    model_path = ARTIFACTS_DIR / "best_lstm_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 切換到評估模式 (關閉 Dropout)

    print(f"🔮 正在對 {len(X_test)} 筆未知的測試集進行預測...")
    
    # ── 4. 進行預測 ──────────────────────────────────────────────────────
    X_test_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad(): # 評估時不需要算梯度，省記憶體
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    # ── 5. 反標準化 (Inverse Transform) ──────────────────────────────────
    # 把模型吐出來的 0.2, -0.5 這種數字，還原回真實的花費金額
    predictions_real = target_scaler.inverse_transform(predictions_scaled)

    # ── 6. 計算最終誤差 ──────────────────────────────────────────────────
    mae   = mean_absolute_error(y_test_raw, predictions_real)
    rmse  = np.sqrt(mean_squared_error(y_test_raw, predictions_real))
    medae = float(np.median(np.abs(predictions_real - y_test_raw)))   # 不受極端值影響

    # 額外幫你算一個 sMAPE (對稱平均絕對百分比誤差)
    smape = np.mean(2.0 * np.abs(predictions_real - y_test_raw) / (np.abs(predictions_real) + np.abs(y_test_raw) + 1e-8)) * 100

    # ── 7. 產出並儲存 txt 報告 (純淨版) ──────────────────────────────────
    report_content = f"""Bi-LSTM Baseline Result
model_name: my_bilstm_baseline
version: v1_independent
pretrained: False (Trained from scratch)
test_mae: {mae:.2f}
test_rmse: {rmse:.2f}
test_medae: {medae:.2f}
test_smape: {smape:.2f}%
feature_type: raw_rolling_and_time
feature_cols: ['daily_expense', 'roll_7d_mean', 'roll_30d_mean', 'dow_sin', 'dow_cos']

==============================================================
            My Bi-LSTM 獨立作戰結果 (越低越好)
==============================================================
  Test MAE   : {mae:.2f}
  Test RMSE  : {rmse:.2f}
  Test MedAE : {medae:.2f}  ← 不受極端值影響
  Test sMAPE : {smape:.2f}%
==============================================================
  💡 若 MAE >> MedAE，代表有少數極端誤差在拉高 MAE
"""

    # 印在終端機給你馬上看
    print(report_content)

    # 寫入 txt 檔案
    report_path = ARTIFACTS_DIR / "my_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"💾 報告已成功儲存，隨時可打開查看： {report_path}")
    print("🎉 恭喜！你的專屬 LSTM 預測管線 (資料 -> 訓練 -> 評估) 大功告成！")

if __name__ == "__main__":
    main()