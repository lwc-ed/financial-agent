import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 導入團隊工具
MY_DIR = Path(__file__).resolve().parent
ML_ROOT = MY_DIR.parent
sys.path.insert(0, str(ML_ROOT))
from output_eval_utils import run_output_evaluation

# ── 1. 模型架構 (保持不變) ──────────────────────────────────────────
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
    print("🚀 [Step 3] 開始進行 Bi-LSTM 期末考...")
    ARTIFACTS_DIR = MY_DIR / "artifacts"
    
    X_test = np.load(ARTIFACTS_DIR / "my_X_test.npy")
    y_test_raw = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy") 
    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    INPUT_SIZE = X_test.shape[2]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = MyBiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_lstm_model.pth", map_location=device, weights_only=True))
    model.eval()

    X_test_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    predictions_real = target_scaler.inverse_transform(predictions_scaled)

    # ── 核心接入：正式評估流程 ────────────────────────────────────────────
    print("\n🏁 [Spec] 正在執行團隊統一評估流程...")
    
    # 1. 讀取 Metadata
    metadata_df = pd.read_csv(ARTIFACTS_DIR / "metadata.csv")

    # 2. 準備輸入資料 (只取 test 部分)
    test_meta = metadata_df[metadata_df['split'] == 'test'].reset_index(drop=True)
    prediction_input_df = pd.DataFrame({
        'user_id': test_meta['user_id'],
        'date': test_meta['date'],
        'y_true': y_test_raw.ravel(),
        'y_pred': predictions_real.ravel()
    })

    # 3. 準備 Metadata 分割表
    split_metadata_df = metadata_df[['user_id', 'date', 'split']]

    # 4. 呼叫共用 evaluator
    run_output_evaluation(
        model_name="bilstm",
        prediction_input_df=prediction_input_df,
        split_metadata_df=split_metadata_df
    )

    print(f"\n✅ 報告已儲存至: {ML_ROOT}/model_outputs/bilstm/")
    print("🎉 bilstm 評估完成！")

if __name__ == "__main__":
    main()