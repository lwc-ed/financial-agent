import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import os

# ── 1. 路徑設定與載入資料 ───────────────────────────────────────────────
# 直接定位到這支程式旁邊的 artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

print("🚀 [Step 2] 開始訓練專屬 LSTM 模型...")
print(f"📂 正在從 {ARTIFACTS_DIR} 載入 3D 彈藥...")

X_train = np.load(ARTIFACTS_DIR / "my_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "my_y_train_scaled.npy")
X_val   = np.load(ARTIFACTS_DIR / "my_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "my_y_val_scaled.npy")

print(f"✅ 載入成功！訓練集大小: {X_train.shape}, 驗證集大小: {X_val.shape}")

# 轉換成 PyTorch 需要的 Tensor 格式
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

# 設定 DataLoader (打包成 Batch 餵給模型)
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── 2. 定義 Bi-LSTM 神經網路架構 ─────────────────────────────────────────
class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(MyBiLSTM, self).__init__()
        # 使用雙向 LSTM (bidirectional=True)，正反向一起看時間規律
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # 因為是雙向，出來的特徵維度會變成 hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # 我們只取最後一天的輸出結果來預測未來
        last_time_step_out = lstm_out[:, -1, :] 
        
        # 通過全連接層 (FC Layers)
        out = self.fc1(last_time_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# 初始化模型參數
INPUT_SIZE = X_train.shape[2]  # 特徵數量 (5個)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"⚙️  使用運算設備: {device}")

model = MyBiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

# ── 3. 訓練設定 (Loss & Optimizer) ──────────────────────────────────────
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10  # 提早停止 (Early Stopping) 的容忍度

# 損失函數：HuberLoss 對極端值 (突然的高消費) 比較有抵抗力
criterion = nn.HuberLoss(delta=1.0) 
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ── 4. 開始訓練迴圈 ──────────────────────────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0
best_model_path = ARTIFACTS_DIR / "best_lstm_model.pth"

print(f"\n🔥 開始訓練 (最多 {EPOCHS} Epochs)...")
print("-" * 50)

for epoch in range(1, EPOCHS + 1):
    # --- 訓練階段 ---
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # --- 驗證階段 ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
            
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # --- Checkpoint 與 Early Stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 存下最好的模型權重
        torch.save(model.state_dict(), best_model_path)
        print("  🌟 找到更好的模型了！已存檔。")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ 驗證誤差連續 {PATIENCE} 次沒下降，提早結束訓練防止過擬合！")
            break

print("-" * 50)
print(f"🎉 [Step 2] 訓練完成！最佳 Val Loss: {best_val_loss:.4f}")
print(f"💾 模型已存於: {best_model_path}")