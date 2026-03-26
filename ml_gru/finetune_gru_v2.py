"""
版本 V2：GRU Finetune（漸進式解凍）
========================================
相對 V1 的改動：

  1. 模型縮小（hidden=64）→ 與 pretrain_gru_v2.pth 對齊
     理由：解決 V1 參數/樣本比過高的過擬合問題

  2. Gradual Unfreezing（漸進式解凍）策略
     理由：V1 全量微調在 epoch 5 就達到最佳，之後立即過擬合。
           漸進式解凍讓模型先學習 task-specific 的輸出層，
           再逐步解鎖 representation 層，類似 ULMFiT 策略。

     Phase 1（epoch 1-25）：只訓練 Attention + FC 層
       - GRU 層保持凍結，保留 Walmart pretrain 的序列理解能力
       - 讓輸出層先適應個人消費的尺度和分布
     Phase 2（epoch 26+）：解凍所有層，LR 除以 3
       - 用更小的 LR 微調整個網路，避免 catastrophic forgetting

  3. Dropout 0.3 → 0.4（對齊 V2 pretrain）
  4. Weight decay 1e-4 → 5e-4
  5. Patience 30 → 20 for Phase 1, 25 for Phase 2（共 45 patience 空間）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle

ARTIFACTS_DIR   = "artificats"
INPUT_SIZE      = 7
BATCH_SIZE      = 16
PHASE1_EPOCHS   = 60    # 只訓練 Attention + FC
PHASE2_EPOCHS   = 140   # 全量微調
LR_PHASE1       = 3e-4  # Phase 1：只有輸出層
LR_PHASE2       = 1e-4  # Phase 2：全量，比 Phase 1 小（穩定遷移）
PATIENCE        = 25
WEIGHT_DECAY    = 5e-4
HUBER_DELTA     = 1.0

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

print("\n📂 載入個人記帳資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
print(f"  X_train : {X_train.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)


class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
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
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


print("\n📦 載入 V2 pretrain 權重...")
checkpoint = torch.load(f"{ARTIFACTS_DIR}/pretrain_gru_v2.pth", map_location=device)
hp         = checkpoint["hyperparams"]
print(f"  Pretrain V2：hidden={hp['hidden_size']}, version={checkpoint.get('version','?')}")

model = GRUWithAttention(
    INPUT_SIZE, hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
model.load_state_dict(checkpoint["model_state"])
print("  ✅ 全部權重成功繼承")


def set_phase(phase: int):
    """
    Phase 1：凍結 GRU，只訓練 Attention + LayerNorm + FC
    Phase 2：解凍所有層
    """
    gru_params    = ["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
                     "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1"]
    for name, param in model.named_parameters():
        if phase == 1 and name in gru_params:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Phase {phase}：可訓練 {trainable:,}/{total:,} 參數 ({trainable/total*100:.1f}%)")


criterion    = nn.HuberLoss(delta=HUBER_DELTA)
best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/finetune_gru_v2.pth"
train_losses, val_losses = [], []
best_hp = hp

# ─────────────────────────────────────────
# Phase 1：凍結 GRU，只訓練輸出層
# ─────────────────────────────────────────
print("\n🔒 Phase 1：凍結 GRU 層，只訓練 Attention + FC...")
set_phase(1)

optimizer_p1 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE1, weight_decay=WEIGHT_DECAY
)
scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p1, T_max=PHASE1_EPOCHS, eta_min=1e-6
)

print(f"\n🚀 Phase 1 開始（{PHASE1_EPOCHS} epochs max, patience={PATIENCE}）...")
print("=" * 65)

phase1_best    = float("inf")
phase1_counter = 0
phase1_stop    = PHASE1_EPOCHS

for epoch in range(1, PHASE1_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer_p1.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_p1.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_loss += criterion(model(X_b), y_b).item()
    val_loss /= len(val_loader)

    scheduler_p1.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr = optimizer_p1.param_groups[0]["lr"]
    print(f"  [P1] Epoch {epoch:3d}/{PHASE1_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        phase1_best      = val_loss
        phase1_counter   = 0
        torch.save({
            "epoch": epoch, "phase": 1,
            "model_state": model.state_dict(),
            "val_loss": best_val_loss, "version": "v2",
            "hyperparams": {
                "input_size": INPUT_SIZE, "hidden_size": hp["hidden_size"],
                "num_layers": hp["num_layers"], "output_size": hp["output_size"],
                "dropout": hp["dropout"],
            }
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        phase1_counter += 1
        if phase1_counter >= PATIENCE:
            print(f"\n⏹️  Phase 1 Early stopping at epoch {epoch}！")
            phase1_stop = epoch
            break

# ─────────────────────────────────────────
# Phase 2：解凍所有層，全量微調
# ─────────────────────────────────────────
print(f"\n🔓 Phase 2：解凍所有層，全量微調（LR={LR_PHASE2}）...")
set_phase(2)

optimizer_p2 = torch.optim.AdamW(
    model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY
)
scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-7
)

patience_counter = 0
print(f"\n🚀 Phase 2 開始（{PHASE2_EPOCHS} epochs max, patience={PATIENCE}）...")
print("=" * 65)

for epoch in range(1, PHASE2_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer_p2.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_p2.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_loss += criterion(model(X_b), y_b).item()
    val_loss /= len(val_loader)

    scheduler_p2.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr = optimizer_p2.param_groups[0]["lr"]
    print(f"  [P2] Epoch {epoch:3d}/{PHASE2_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch": phase1_stop + epoch, "phase": 2,
            "model_state": model.state_dict(),
            "val_loss": best_val_loss, "version": "v2",
            "hyperparams": {
                "input_size": INPUT_SIZE, "hidden_size": hp["hidden_size"],
                "num_layers": hp["num_layers"], "output_size": hp["output_size"],
                "dropout": hp["dropout"],
            }
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Phase 2 Early stopping！")
            break

print("=" * 65)
print(f"\n🎉 V2 Finetune 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/finetune_history_v2.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 predict_v2.py")
