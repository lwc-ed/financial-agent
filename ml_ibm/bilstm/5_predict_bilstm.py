import glob, sys, pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bilstm"
DATA_DIR = MY_DIR.parent / "bigru_TL_alignment" / "artifacts_bigru_tl"

# 接入團隊共用工具
ML_UTILS_DIR = MY_DIR.parents[1] / "ml" 
sys.path.insert(0, str(MY_DIR))
sys.path.insert(0, str(ML_UTILS_DIR))
from model_bilstm import MyBiLSTM
from output_eval_utils import run_output_evaluation

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

print("📂 執行 Bi-LSTM (IBM TL) 最終評估...")
X_test = np.load(DATA_DIR / "personal_X_test.npy")
y_test_raw = np.load(DATA_DIR / "personal_y_test_raw.npy")
with open(DATA_DIR / "personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# 載入 7 個 Seed 並取預測平均值 (Ensemble)
SEEDS = [42, 123, 777, 456, 789, 999, 2024]
all_preds = []
for seed in SEEDS:
    model = MyBiLSTM(X_test.shape[2], 64, 2, 1).to(device)
    ckpt = torch.load(ARTIFACTS_DIR / f"finetune_bilstm_seed{seed}.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X_test).to(device)).cpu().numpy()
        all_preds.append(target_scaler.inverse_transform(p))

avg_preds = np.mean(all_preds, axis=0)

# 準備 Metadata 進行正式評估
metadata_df = pd.read_csv(DATA_DIR / "metadata.csv")
test_meta = metadata_df[metadata_df['split'] == 'test'].reset_index(drop=True)

prediction_input_df = pd.DataFrame({
    'user_id': test_meta['user_id'],
    'date': test_meta['date'],
    'y_true': y_test_raw.ravel(),
    'y_pred': avg_preds.ravel()
})

run_output_evaluation(
    model_name="bilstm",
    prediction_input_df=prediction_input_df,
    split_metadata_df=metadata_df[['user_id', 'date', 'split']],
    output_root=MY_DIR.parent / "model_outputs"
)

print(f"\n✅ 報表已儲存至: {MY_DIR.parent}/model_outputs/bilstm/")
print("🎉 Bi-LSTM 期末考結束！")