import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

MY_DIR = Path(__file__).resolve().parent
DATA_DIR = MY_DIR.parent / "bigru_TL_alignment" / "artifacts_bigru_tl"
SAVE_DIR = MY_DIR / "artifacts_bilstm"

sys.path.insert(0, str(MY_DIR))
from model_bilstm import MyBiLSTM

SEEDS = [42, 123, 777, 456, 789, 999, 2024] # 正式規格要求的 7 個種子
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 載入個人資料
X_train = np.load(DATA_DIR / "personal_X_train.npy")
y_train = np.load(DATA_DIR / "personal_y_train.npy")
X_val   = np.load(DATA_DIR / "personal_X_val.npy")
y_val   = np.load(DATA_DIR / "personal_y_val.npy")

print(f"🚀 開始加載 IBM 大腦進行【個人微調】...")
for seed in SEEDS:
    save_path = SAVE_DIR / f"finetune_bilstm_seed{seed}.pth"
    torch.manual_seed(seed)
    
    # 【關鍵】載入剛才重寫後的 IBM 預訓練模型
    ckpt = torch.load(SAVE_DIR / "pretrain_bilstm.pth", map_location=device, weights_only=True)
    model = MyBiLSTM(X_train.shape[2], 64, 2, 1).to(device)
    model.load_state_dict(ckpt["model_state"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.HuberLoss()
    
    # 微調過程
    model.train()
    for _ in range(30):
        for X_b, y_b in DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(X_b.to(device)), y_b.to(device))
            loss.backward()
            optimizer.step()
    
    torch.save({"model_state": model.state_dict()}, save_path)
    print(f"✅ Seed {seed} 微調完成")

print("\n🎉 Bi-LSTM 遷移學習完成！現在可以產出正式報表了。")