"""
gru_lower_model_peruser_finetune/finetune_peruser.py
======================================================
Step 2：對每個使用者個別 fine-tune 全域模型。

策略：
  Phase 1（凍結 GRU，只訓練 FC head）：
    - LR = 1e-3，epochs = 30
    - 讓 FC 快速適應該使用者的消費量級
  Phase 2（解凍所有層）：
    - LR = 1e-4，epochs = 50
    - 細調 GRU 以捕捉使用者特有的時序模式

為什麼要凍結/解凍？
  - 每個使用者約只有 ~178 筆訓練資料
  - 一次性解凍所有層容易毀掉全域模型學到的通用特徵
  - Phase 1 先讓 FC 輸出對齊使用者量級，再整體微調更穩定

每個使用者的模型儲存至 artifacts/user_{user_id}.pth
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from datetime import datetime

ARTIFACTS_DIR  = "artifacts"
INPUT_SIZE     = 7
OUTPUT_SIZE    = 1

# Fine-tune 超參數
PHASE1_EPOCHS  = 30    # 凍結 GRU，只訓練 FC
PHASE2_EPOCHS  = 50    # 解凍所有層
PHASE1_LR      = 1e-3
PHASE2_LR      = 1e-4
WEIGHT_DECAY   = 5e-4
BATCH_SIZE     = 16
PATIENCE       = 15    # per-phase early stopping

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")


class SmallGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
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

    def freeze_gru(self):
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.attention.parameters():
            p.requires_grad = False
        for p in self.layer_norm.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def train_phase(model, X_tr, y_tr, X_va, y_va, lr, epochs, patience, label):
    """通用訓練函式，回傳最佳 val_loss"""
    from torch.utils.data import DataLoader, TensorDataset

    loader_tr = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=min(BATCH_SIZE, len(X_tr)), shuffle=True)
    loader_va = DataLoader(
        TensorDataset(torch.tensor(X_va), torch.tensor(y_va)),
        batch_size=min(BATCH_SIZE, len(X_va)), shuffle=False)

    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=WEIGHT_DECAY)
    huber = nn.HuberLoss(delta=1.0)
    mse   = nn.MSELoss()
    crit  = lambda p, t: 0.7 * huber(p, t) + 0.3 * mse(p, t)

    best_loss    = float("inf")
    best_state   = None
    no_improve   = 0

    for ep in range(1, epochs + 1):
        model.train()
        for X_b, y_b in loader_tr:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in loader_va:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += crit(model(X_b), y_b).item()
        val_loss /= len(loader_va)

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


# ─────────────────────────────────────────
# 載入全域模型 & per-user 資料
# ─────────────────────────────────────────
print("\n📦 載入全域模型...")
ckpt = torch.load(f"{ARTIFACTS_DIR}/global_model.pth", map_location=device)
hp   = ckpt["hyperparams"]

with open(f"{ARTIFACTS_DIR}/per_user_data.pkl", "rb") as f:
    per_user_data = pickle.load(f)
with open(f"{ARTIFACTS_DIR}/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

user_ids = sorted(per_user_data.keys(), key=str)
print(f"  全域模型 val_loss: {ckpt['val_loss']:.6f}")
print(f"  需要 fine-tune 的使用者數: {len(user_ids)}")

# ─────────────────────────────────────────
# Per-user fine-tune
# ─────────────────────────────────────────
user_metrics = {}

for i, uid in enumerate(user_ids):
    d = per_user_data[uid]
    X_tr, y_tr = d["X_train_sc"], d["y_train_sc"]
    X_va, y_va = d["X_val_sc"],   d["y_val_sc"]
    X_te, y_te = d["X_test_sc"],  d["y_test_sc"]

    print(f"\n[{i+1}/{len(user_ids)}] 使用者 {uid}  "
          f"train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    if len(X_va) == 0:
        print(f"  ⚠️  val 集為空，跳過（保留全域模型預測）")
        # 直接複製全域模型給這個使用者
        torch.save(ckpt, f"{ARTIFACTS_DIR}/user_{uid}.pth")
        continue

    # 載入全域模型（每個 user 從全域模型出發，互不干擾）
    model = SmallGRU(hp["input_size"], hp["hidden_size"], hp["output_size"], hp["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Phase 1：凍結 GRU，只訓練 FC head
    model.freeze_gru()
    loss_p1 = train_phase(model, X_tr, y_tr, X_va, y_va,
                           PHASE1_LR, PHASE1_EPOCHS, PATIENCE, "Phase1")
    print(f"  Phase1 (FC only) best val_loss: {loss_p1:.6f}")

    # Phase 2：解凍所有層，細調
    model.unfreeze_all()
    loss_p2 = train_phase(model, X_tr, y_tr, X_va, y_va,
                           PHASE2_LR, PHASE2_EPOCHS, PATIENCE, "Phase2")
    print(f"  Phase2 (全層)    best val_loss: {loss_p2:.6f}")

    # 計算 test MAE（scaled space → 反轉換）
    model.eval()
    with torch.no_grad():
        pred_sc = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    pred_raw = target_scaler.inverse_transform(pred_sc).flatten()
    true_raw = target_scaler.inverse_transform(y_te).flatten()
    mae = float(np.mean(np.abs(pred_raw - true_raw)))
    print(f"  Test MAE: {mae:,.2f}")

    user_metrics[str(uid)] = {
        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
        "phase1_val_loss": round(loss_p1, 6),
        "phase2_val_loss": round(loss_p2, 6),
        "test_mae": round(mae, 2),
    }

    torch.save({
        "model_state": model.state_dict(),
        "hyperparams": hp,
        "user_id": str(uid),
        "phase1_val_loss": loss_p1,
        "phase2_val_loss": loss_p2,
    }, f"{ARTIFACTS_DIR}/user_{uid}.pth")

# ─────────────────────────────────────────
# 儲存總覽
# ─────────────────────────────────────────
maes = [v["test_mae"] for v in user_metrics.values() if "test_mae" in v]
avg_mae = float(np.mean(maes)) if maes else 0.0

print(f"\n{'='*55}")
print(f"  Per-user fine-tune 完成！")
print(f"  平均 Test MAE（各使用者）：{avg_mae:,.2f}")
print(f"  使用者數：{len(user_metrics)}")
print(f"{'='*55}")

summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "avg_test_mae": round(avg_mae, 2),
    "user_metrics": user_metrics,
}
with open(f"{ARTIFACTS_DIR}/peruser_finetune_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n下一步：python predict.py")
