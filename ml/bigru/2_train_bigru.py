import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import os

# ── 0. 環境與裝置設定 ──────────────────────────────────────────────────
# 檢查有沒有 GPU 可以用
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"🖥️ 目前使用的運算裝置: {device}")

# ── 1. 路徑設定與載入資料 ───────────────────────────────────────────────
# 直接定位到這支程式旁邊的 artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

print("🚀 [Step 2] 開始訓練標準版 Bi-GRU 模型...")
print(f"📂 正在從 {ARTIFACTS_DIR} 載入 3D 彈藥...")

# 確保你的 artifacts 資料夾裡有這些檔案
X_train = np.load(ARTIFACTS_DIR / "my_X_train.npy").astype(np.float32)
y_train = np.load(ARTIFACTS_DIR / "my_y_train_scaled.npy").astype(np.float32)
X_val   = np.load(ARTIFACTS_DIR / "my_X_val.npy").astype(np.float32)
y_val   = np.load(ARTIFACTS_DIR / "my_y_val_scaled.npy").astype(np.float32)

print(f"✅ 載入成功！訓練集大小: {X_train.shape}, 驗證集大小: {X_val.shape}")

# 轉換成 PyTorch 需要的 Tensor 格式
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

# 設定 DataLoader (打包成 Batch 餵給模型)
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── 2. 定義 Bi-GRU 模型 ────────────────────────────────────────────────
class MyBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(MyBiGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 🔑 這裡換成了 nn.GRU
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 啟動雙向
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size) # 雙向所以乘 2

    def forward(self, x):
        out, _ = self.gru(x) 
        out = out[:, -1, :] # 只取最後一個時間步
        out = self.dropout(out)
        predictions = self.fc(out)
        return predictions

# 實體化模型並丟到 GPU/CPU
INPUT_SIZE = X_train.shape[2]  # 自動抓取特徵維度 (不用手動設定 10 了)
model = MyBiGRU(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, output_size=1).to(device)

# ── 3. 訓練設定 (Loss & Optimizer) ──────────────────────────────────────
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10  # 提早停止 (Early Stopping) 的容忍度

# 損失函數：HuberLoss 對極端值比較有抵抗力
criterion = nn.HuberLoss(delta=1.0) 
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ── 4. 開始訓練迴圈 ──────────────────────────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0

# 🔑 這裡存檔名字改掉，避免覆蓋之前的 LSTM！
best_model_path = ARTIFACTS_DIR / "best_standard_bigru.pth"

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
        torch.save(model.state_dict(), best_model_path)
        print("   🌟 找到更好的模型了！已存檔。")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ 驗證誤差連續 {PATIENCE} 次沒下降，提早結束訓練防止過擬合！")
            break

print("-" * 50)
print(f"🎉 [Step 2] 訓練完成！最佳 Val Loss: {best_val_loss:.4f}")
print(f"💾 模型已存於: {best_model_path}")