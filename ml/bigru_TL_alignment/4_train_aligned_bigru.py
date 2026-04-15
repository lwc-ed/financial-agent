import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

# ── 0. 環境與裝置設定 ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 目前使用的運算裝置: {device}")

# ── 1. 路徑設定與載入「Alignment 版」資料 ──────────────────────────────
# 🔑 這次直接讀取你自己剛剛熱騰騰生出來的 artifacts_aligned！
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts_aligned"

print("🚀 [Step 4] 開始訓練 Alignment 版 Bi-GRU 模型...")
print(f"📂 正在從 {ARTIFACTS_DIR} 載入對齊特徵彈藥...")

try:
    X_train = np.load(ARTIFACTS_DIR / "personal_aligned_X_train.npy").astype(np.float32)
    y_train = np.load(ARTIFACTS_DIR / "personal_aligned_y_train.npy").astype(np.float32)
    X_val   = np.load(ARTIFACTS_DIR / "personal_aligned_X_val.npy").astype(np.float32)
    y_val   = np.load(ARTIFACTS_DIR / "personal_aligned_y_val.npy").astype(np.float32)
    print(f"✅ 載入成功！訓練集大小: {X_train.shape}, 驗證集大小: {X_val.shape}")
except FileNotFoundError:
    print(f"❌ 找不到檔案！請確認你剛剛有沒有成功跑完 Step 1。")
    exit()

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── 2. 定義 Bi-GRU 模型 ────────────────────────────────
class MyBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(MyBiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, 
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.gru(x) 
        out = out[:, -1, :] 
        out = self.dropout(out)
        return self.fc(out)

INPUT_SIZE = X_train.shape[2]  # Alignment 特徵數是 7
model = MyBiGRU(input_size=INPUT_SIZE, hidden_size=64, num_layers=2, output_size=1).to(device)

# ── 3. 訓練設定 ────────────────────────────────────────────────────────
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10 

criterion = nn.HuberLoss(delta=1.0) 
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ── 4. 開始訓練迴圈 ────────────────────────────────────────────────────
best_val_loss = float("inf")
patience_counter = 0

# 🔑 存檔在你的 artifacts_aligned 資料夾裡！
best_model_path = ARTIFACTS_DIR / "best_aligned_bigru.pth"

print(f"\n🔥 開始訓練 (最多 {EPOCHS} Epochs)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_loss += criterion(model(X_batch), y_batch).item()
            
    val_loss /= len(val_loader)
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ 驗證誤差連續 {PATIENCE} 次沒下降，提早結束！")
            break

print(f"🎉 訓練完成！模型已存於: {best_model_path}")