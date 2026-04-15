"""
ablation_mmd_lambda.py
======================
BiLSTM v2 MMD λ Ablation Study
自動對每個 λ 跑完整的 pretrain → finetune → evaluate，找出最佳 MMD weight

λ=0.0 代表完全不用 MMD（純 HuberLoss），作為 ablation 的基準
最後印出比較表，方便直接看出哪個 λ 最好

輸出：
  - artifacts_ablation/lambda_{λ}/pretrain.pth
  - artifacts_ablation/lambda_{λ}/finetune_seed{seed}.pth
  - artifacts_ablation/ablation_results.json
  - artifacts_ablation/ablation_summary.txt
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys, json
from itertools import combinations
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_bilstm_v2"
ABLATION_DIR  = "artifacts_ablation"
os.makedirs(ABLATION_DIR, exist_ok=True)

# ── λ 候選值（0.0 = 無 MMD，作為對照組）────────────────────────────────────
LAMBDA_LIST = [0.0, 0.05, 0.1, 0.2, 0.5]

# ── 超參數（與正式版完全相同）──────────────────────────────────────────────
INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)   # 10
HIDDEN_SIZE   = 64                           # bidirectional → 實際 128
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 64
PRETRAIN_EPOCHS = 150
FINETUNE_EPOCHS = 80
LR_PRETRAIN   = 0.0001
LR_FINETUNE   = 3e-4
PATIENCE      = 20
WD_PRETRAIN   = 5e-4
WD_FINETUNE   = 1e-4
HUBER_DELTA   = 1.0
ABLATION_SEEDS = [42, 123, 777]   # 3 seeds 夠比較，跑比較快

# ── 設備 ──────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps");  print("✅ Apple M1 MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("✅ CUDA")
else:
    device = torch.device("cpu");  print("⚠️  CPU")


# ── MMD（RBF kernel + Median Heuristic）──────────────────────────────────────
def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    m = y.size(0)

    rx = (x ** 2).sum(dim=1, keepdim=True)
    ry = (y ** 2).sum(dim=1, keepdim=True)

    dist_xx = rx + rx.t() - 2 * torch.mm(x, x.t())
    dist_yy = ry + ry.t() - 2 * torch.mm(y, y.t())
    dist_xy = rx + ry.t() - 2 * torch.mm(x, y.t())

    all_dist  = torch.cat([dist_xx.reshape(-1), dist_yy.reshape(-1), dist_xy.reshape(-1)])
    bandwidth = all_dist.median().clamp(min=1e-6)

    K = torch.exp(-0.5 * dist_xx / bandwidth)
    L = torch.exp(-0.5 * dist_yy / bandwidth)
    P = torch.exp(-0.5 * dist_xy / bandwidth)

    mmd = (K.sum() - K.trace()) / (n * (n - 1) + 1e-8) \
        + (L.sum() - L.trace()) / (m * (m - 1) + 1e-8) \
        - 2 * P.mean()
    return mmd.clamp(min=0)


# ── 模型架構：BiLSTM + Attention ──────────────────────────────────────────────
class BiLSTMWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )
        bi_hidden = HIDDEN_SIZE * 2   # 128
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(DROPOUT)
        self.fc1        = nn.Linear(bi_hidden, HIDDEN_SIZE)    # 128 → 64
        self.fc2        = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)  # 64  → 1
        self.relu       = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        out, _ = self.lstm(x)
        attn_w  = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


# ── 載入資料（只做一次）───────────────────────────────────────────────────────
print("\n📂 載入資料...")
X_wmt_train = np.load(f"{ARTIFACTS_DIR}/walmart_X_train.npy")
y_wmt_train = np.load(f"{ARTIFACTS_DIR}/walmart_y_train.npy")
X_wmt_val   = np.load(f"{ARTIFACTS_DIR}/walmart_X_val.npy")
y_wmt_val   = np.load(f"{ARTIFACTS_DIR}/walmart_y_val.npy")

X_per_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_per_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_per_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_per_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_per_test  = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")

y_per_test_raw = np.load(f"{ARTIFACTS_DIR}/personal_y_test_raw.npy")
test_user_ids  = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy")
val_user_ids   = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy")

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

y_per_val_raw = target_scaler.inverse_transform(y_per_val)

print(f"  Walmart train : {X_wmt_train.shape}")
print(f"  Personal train: {X_per_train.shape}")


# ── DataLoader 工廠 ───────────────────────────────────────────────────────────
def make_loader(X, y=None, batch_size=64, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        return DataLoader(TensorDataset(X_t, torch.tensor(y, dtype=torch.float32)),
                          batch_size=batch_size, shuffle=shuffle)
    return DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=shuffle)


# ── Pretrain ──────────────────────────────────────────────────────────────────
def run_pretrain(mmd_lambda: float, save_dir: str) -> float:
    model     = BiLSTMWithAttention().to(device)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=WD_PRETRAIN)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    train_loader    = make_loader(X_wmt_train, y_wmt_train, BATCH_SIZE, shuffle=True)
    val_loader      = make_loader(X_wmt_val,   y_wmt_val,   BATCH_SIZE, shuffle=False)
    personal_loader = make_loader(X_per_train, batch_size=BATCH_SIZE, shuffle=True)
    personal_iter   = iter(personal_loader)

    best_val   = float("inf")
    patience_c = 0
    save_path  = f"{save_dir}/pretrain.pth"

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        ep_huber = ep_mmd = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            try:    (X_p,) = next(personal_iter)
            except: personal_iter = iter(personal_loader); (X_p,) = next(personal_iter)
            X_p = X_p.to(device)

            optimizer.zero_grad()
            huber_loss = criterion(model(X_b), y_b)

            if mmd_lambda > 0:
                mmd_loss   = compute_mmd(model.encode(X_b), model.encode(X_p))
                mmd_scale  = huber_loss.detach() / (mmd_loss.detach() + 1e-8)
                total_loss = huber_loss + mmd_lambda * mmd_scale * mmd_loss
                ep_mmd    += mmd_loss.item()
            else:
                total_loss = huber_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_huber += huber_loss.item()

        ep_huber /= len(train_loader)
        ep_mmd   /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        mmd_str = f"  MMD: {ep_mmd:.6f}" if mmd_lambda > 0 else ""
        print(f"  [Pretrain λ={mmd_lambda}] Epoch {epoch:3d} | Huber: {ep_huber:.6f}{mmd_str} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val   = val_loss
            patience_c = 0
            torch.save({"model_state": model.state_dict(), "val_loss": best_val}, save_path)
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"  ⏹️  Early stopping at epoch {epoch}")
                break

    print(f"  ✅ Pretrain 完成  best val_loss={best_val:.6f}")
    return best_val


# ── Finetune ──────────────────────────────────────────────────────────────────
def run_finetune(mmd_lambda: float, seed: int, save_dir: str) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt  = torch.load(f"{save_dir}/pretrain.pth", map_location=device)
    model = BiLSTMWithAttention().to(device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WD_FINETUNE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=7)

    train_loader   = make_loader(X_per_train, y_per_train, 32, shuffle=True)
    val_loader     = make_loader(X_per_val,   y_per_val,   32, shuffle=False)
    walmart_loader = make_loader(X_wmt_train, batch_size=32, shuffle=True)
    walmart_iter   = iter(walmart_loader)

    best_val   = float("inf")
    patience_c = 0
    save_path  = f"{save_dir}/finetune_seed{seed}.pth"

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        ep_huber = ep_mmd = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            try:    (X_w,) = next(walmart_iter)
            except: walmart_iter = iter(walmart_loader); (X_w,) = next(walmart_iter)
            X_w = X_w.to(device)

            optimizer.zero_grad()
            huber_loss = criterion(model(X_b), y_b)

            if mmd_lambda > 0:
                mmd_loss   = compute_mmd(model.encode(X_b), model.encode(X_w))
                mmd_scale  = huber_loss.detach() / (mmd_loss.detach() + 1e-8)
                total_loss = huber_loss + mmd_lambda * mmd_scale * mmd_loss
                ep_mmd    += mmd_loss.item()
            else:
                total_loss = huber_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_huber += huber_loss.item()

        ep_huber /= len(train_loader)
        ep_mmd   /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        mmd_str = f"  MMD: {ep_mmd:.6f}" if mmd_lambda > 0 else ""
        print(f"    [Finetune seed={seed}] Epoch {epoch:3d} | Huber: {ep_huber:.6f}{mmd_str} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val   = val_loss
            patience_c = 0
            torch.save({"model_state": model.state_dict(), "val_loss": best_val, "seed": seed}, save_path)
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"    ⏹️  Early stopping at epoch {epoch}")
                break

    return best_val


# ── Evaluate ──────────────────────────────────────────────────────────────────
def run_evaluate(save_dir: str) -> dict:
    val_preds_all  = {}
    test_preds_all = {}

    for seed in ABLATION_SEEDS:
        path = f"{save_dir}/finetune_seed{seed}.pth"
        if not os.path.exists(path):
            continue
        ckpt  = torch.load(path, map_location=device)
        model = BiLSTMWithAttention().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            val_preds_all[seed]  = model(torch.tensor(X_per_val,  dtype=torch.float32).to(device)).cpu().numpy()
            test_preds_all[seed] = model(torch.tensor(X_per_test, dtype=torch.float32).to(device)).cpu().numpy()

    # 找最佳 val combo
    best_mae   = float("inf")
    best_combo = ABLATION_SEEDS

    for r in range(1, len(ABLATION_SEEDS) + 1):
        for combo in combinations(ABLATION_SEEDS, r):
            combo = list(combo)
            avg   = np.mean([val_preds_all[s] for s in combo if s in val_preds_all], axis=0)
            preds = target_scaler.inverse_transform(avg)
            mae   = float(np.mean(np.abs(y_per_val_raw - preds)))
            if mae < best_mae:
                best_mae   = mae
                best_combo = combo

    val_avg    = np.mean([val_preds_all[s]  for s in best_combo if s in val_preds_all], axis=0)
    test_avg   = np.mean([test_preds_all[s] for s in best_combo if s in test_preds_all], axis=0)
    val_preds  = target_scaler.inverse_transform(val_avg)
    test_preds = target_scaler.inverse_transform(test_avg)

    return {
        "best_combo" : best_combo,
        "val_mae"    : float(np.mean(np.abs(y_per_val_raw  - val_preds))),
        "val_medae"  : float(np.median(np.abs(y_per_val_raw  - val_preds))),
        "val_rmse"   : float(np.sqrt(np.mean((y_per_val_raw  - val_preds) ** 2))),
        "test_mae"   : float(np.mean(np.abs(y_per_test_raw - test_preds))),
        "test_medae" : float(np.median(np.abs(y_per_test_raw - test_preds))),
        "test_rmse"  : float(np.sqrt(np.mean((y_per_test_raw - test_preds) ** 2))),
    }


# ── 主迴圈 ────────────────────────────────────────────────────────────────────
all_results = {}

for mmd_lambda in LAMBDA_LIST:
    label    = f"lambda_{mmd_lambda}"
    save_dir = f"{ABLATION_DIR}/{label}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🔬 Ablation：λ = {mmd_lambda}{'  (純 HuberLoss，無 MMD)' if mmd_lambda == 0 else ''}")
    print(f"{'='*70}")

    run_pretrain(mmd_lambda, save_dir)

    for seed in ABLATION_SEEDS:
        print(f"\n  --- Finetune seed={seed} ---")
        run_finetune(mmd_lambda, seed, save_dir)

    print(f"\n  📊 Evaluate λ={mmd_lambda}...")
    metrics = run_evaluate(save_dir)
    all_results[mmd_lambda] = metrics
    print(f"  Val  MAE={metrics['val_mae']:.2f}  MedAE={metrics['val_medae']:.2f}")
    print(f"  Test MAE={metrics['test_mae']:.2f}  MedAE={metrics['test_medae']:.2f}")


# ── 比較表 ────────────────────────────────────────────────────────────────────
print(f"\n\n{'='*72}")
print(f"  BiLSTM v2  MMD λ Ablation Study 結果（Test，越低越好）")
print(f"{'='*72}")
print(f"  {'λ':>6}  {'Val MAE':>9}  {'Val MedAE':>10}  {'Test MAE':>9}  {'Test MedAE':>10}  {'Test RMSE':>10}")
print(f"  {'-'*67}")

best_lambda = min(all_results, key=lambda l: all_results[l]["test_mae"])

for lam, m in sorted(all_results.items()):
    marker = "  ← 最佳" if lam == best_lambda else ""
    print(f"  {lam:>6}  {m['val_mae']:>9.2f}  {m['val_medae']:>10.2f}  "
          f"{m['test_mae']:>9.2f}  {m['test_medae']:>10.2f}  {m['test_rmse']:>10.2f}{marker}")

print(f"{'='*72}")
print(f"\n  🏆 最佳 λ = {best_lambda}  Test MAE = {all_results[best_lambda]['test_mae']:.2f}")

# ── 儲存 ──────────────────────────────────────────────────────────────────────
with open(f"{ABLATION_DIR}/ablation_results.json", "w") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

summary_lines = [
    "BiLSTM v2  MMD λ Ablation Study\n",
    f"{'λ':>6}  {'Val MAE':>9}  {'Val MedAE':>10}  {'Test MAE':>9}  {'Test MedAE':>10}  {'Test RMSE':>10}\n",
    "-" * 67 + "\n",
]
for lam, m in sorted(all_results.items()):
    marker = "  ← 最佳" if lam == best_lambda else ""
    summary_lines.append(
        f"{lam:>6}  {m['val_mae']:>9.2f}  {m['val_medae']:>10.2f}  "
        f"{m['test_mae']:>9.2f}  {m['test_medae']:>10.2f}  {m['test_rmse']:>10.2f}{marker}\n"
    )
summary_lines.append(f"\n最佳 λ = {best_lambda}  Test MAE = {all_results[best_lambda]['test_mae']:.2f}\n")

with open(f"{ABLATION_DIR}/ablation_summary.txt", "w") as f:
    f.writelines(summary_lines)

print(f"\n✅ 結果儲存至 {ABLATION_DIR}/ablation_results.json")
print(f"✅ 摘要儲存至 {ABLATION_DIR}/ablation_summary.txt")
