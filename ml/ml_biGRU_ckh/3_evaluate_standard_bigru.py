import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── 1. 重新定義相同的模型架構 ──────────────────────────────────────────
class MyBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(MyBiGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.gru(x) 
        out = out[:, -1, :]
        out = self.dropout(out)
        predictions = self.fc(out)
        return predictions

def main():
    print("🚀 [Step 3] 開始進行 Bi-GRU 模型期末考 (最終評估)...")

    # ── 2. 路徑設定與載入測試資料 ────────────────────────────────────────
    ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
    
    X_test = np.load(ARTIFACTS_DIR / "my_X_test.npy").astype(np.float32)
    y_test_raw = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy").astype(np.float32)
    
    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    # ── 3. 載入訓練好的模型權重 ──────────────────────────────────────────
    INPUT_SIZE = X_test.shape[2]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyBiGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    
    model_path = ARTIFACTS_DIR / "best_standard_bigru.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"🔮 正在對 {len(X_test)} 筆未知的測試集進行預測...")
    
    # ── 4. 進行預測 ──────────────────────────────────────────────────────
    X_test_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    # ── 5. 反標準化 (Inverse Transform) ──────────────────────────────────
    predictions_real = target_scaler.inverse_transform(predictions_scaled)

    # ── 6. 計算最終誤差 ──────────────────────────────────────────────────
    mae = mean_absolute_error(y_test_raw, predictions_real)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_real))
    smape = np.mean(2.0 * np.abs(predictions_real - y_test_raw) / (np.abs(predictions_real) + np.abs(y_test_raw) + 1e-8)) * 100

    # ── 7. 產出並儲存 txt 報告 ──────────────────────────────────
    # 注意看這裡！下面這段就是剛剛報錯的地方，最後一行的 """ 非常重要！
    report_content = f"""Bi-GRU Baseline Result
model_name: my_bigru_standard
version: v1_standard
pretrained: False (Trained from scratch)
test_mae: {mae:.2f}
test_rmse: {rmse:.2f}
test_smape: {smape:.2f}%

==============================================================
            My Standard Bi-GRU 獨立作戰結果 (越低越好)
==============================================================
  Test MAE   : {mae:.2f} 
  Test RMSE  : {rmse:.2f}
  Test sMAPE : {smape:.2f}%
==============================================================
"""

    print(report_content)

    report_path = ARTIFACTS_DIR / "standard_bigru_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"💾 報告已成功儲存，隨時可打開查看： {report_path}")
    print("🎉 恭喜！標準版 Bi-GRU 評估完成，準備迎接 Alignment 最終挑戰！")

if __name__ == "__main__":
    main()