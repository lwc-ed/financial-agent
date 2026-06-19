"""
Step 4：GRU Finetune（IBM TL，Multi-task）
==========================================
載入 IBM Pretrained GRU，在個人資料上做 finetune
Loss = HuberLoss + MT_ALPHA × FocalLoss（4-class risk level）
  - 回歸頭：預測未來 7 天花費金額
  - 分類頭：同時預測 risk level（no_alarm / low / mid / high）
  - class-weighted Focal Loss（gamma=2）處理少數 class
Ensemble 多 seeds
輸出：
  - finetune_aligned_gru_seed{seed}.pth
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "ml")))
from alignment_utils import ALIGNED_FEATURE_COLS
from output_eval_utils import (
    _prepare_prediction_input,
    _prepare_split_metadata,
    _prepare_transactions,
    build_spent_mtd_lookup,
    compute_4class_risk_metrics,
    compute_binary_alarm_metrics,
    compute_future_available_7d,
    compute_monthly_available_cash,
    compute_regression_metrics,
    compute_risk_ratio,
    lookup_spent_mtd,
    risk_ratio_to_alarm,
    risk_ratio_to_level,
)

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_aligned"

INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
NUM_CLASSES   = 4
BATCH_SIZE    = 32
EPOCHS        = 45
LEARNING_RATE = 1e-3
ENCODER_LR    = 1e-5   # phase 2：unfreeze 後 encoder 用小 LR
FREEZE_EPOCHS = 20     # 前 N epoch freeze encoder，讓 regression head 先穩定
PATIENCE      = 8
WEIGHT_DECAY  = 1e-5
HUBER_DELTA   = 1.0
MSE_WEIGHT    = 0.10
MT_ALPHA      = 0.00  # 正式 F1 由 regression 金額推得，避免分類頭牽動金額預測
RISK_RATIO_ALPHA = 0.00
ORDINAL_ALPHA    = 0.00
BOUNDARY_TEMP    = 0.12
FOCAL_GAMMA   = 2.0   # focal loss gamma
SEEDS = [
    42, 123, 777, 456, 789, 999, 2024,
    0, 7, 13, 21, 100, 314, 1234, 9999,
    11, 22, 33, 44, 55, 66, 77, 88, 99,
    111, 222, 333, 444, 555, 666
]
SELECTION_EXCLUDE_USER_IDS = {"user14"}

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
else:
    device = torch.device("cpu")

print("📂 載入個人資料...")
X_train      = np.load(ARTIFACTS_DIR / "personal_aligned_X_train.npy")
y_train      = np.load(ARTIFACTS_DIR / "personal_aligned_y_train.npy")
X_val        = np.load(ARTIFACTS_DIR / "personal_aligned_X_val.npy")
y_val        = np.load(ARTIFACTS_DIR / "personal_aligned_y_val.npy")
train_labels = np.load(ARTIFACTS_DIR / "personal_aligned_y_train_risk_labels.npy")
val_labels   = np.load(ARTIFACTS_DIR / "personal_aligned_y_val_risk_labels.npy")
train_user_ids = np.load(ARTIFACTS_DIR / "personal_aligned_train_user_ids.npy")
val_user_ids   = np.load(ARTIFACTS_DIR / "personal_aligned_val_user_ids.npy")
test_user_ids  = np.load(ARTIFACTS_DIR / "personal_aligned_test_user_ids.npy")
train_dates = np.load(ARTIFACTS_DIR / "personal_aligned_train_dates.npy")
val_dates   = np.load(ARTIFACTS_DIR / "personal_aligned_val_dates.npy")
test_dates  = np.load(ARTIFACTS_DIR / "personal_aligned_test_dates.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")
print(f"  Train risk 分佈: {dict(sorted(Counter(train_labels.tolist()).items()))}")

import pickle
with open(ARTIFACTS_DIR / "personal_aligned_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
y_train_raw = target_scaler.inverse_transform(y_train)
y_val_raw = target_scaler.inverse_transform(y_val)

train_prediction_input_df = pd.DataFrame({
    "user_id": train_user_ids,
    "date": pd.to_datetime(train_dates),
    "y_true": y_train_raw.ravel(),
    "y_pred": y_train_raw.ravel(),
})
val_prediction_input_df = pd.DataFrame({
    "user_id": val_user_ids,
    "date": pd.to_datetime(val_dates),
    "y_true": y_val_raw.ravel(),
    "y_pred": y_val_raw.ravel(),
})
split_metadata_df = pd.concat([
    pd.DataFrame({"user_id": train_user_ids, "date": pd.to_datetime(train_dates), "split": "train"}),
    pd.DataFrame({"user_id": val_user_ids,   "date": pd.to_datetime(val_dates),   "split": "val"}),
    pd.DataFrame({"user_id": test_user_ids,  "date": pd.to_datetime(test_dates),  "split": "test"}),
], ignore_index=True)


def build_metric_context(prediction_input_df: pd.DataFrame, split_df: pd.DataFrame) -> dict:
    pred_df = _prepare_prediction_input(prediction_input_df)
    split_df = _prepare_split_metadata(split_df)
    raw_txn_df = _prepare_transactions(None)
    monthly_cash_df = compute_monthly_available_cash(raw_txn_df, split_df)
    spent_lookup = build_spent_mtd_lookup(raw_txn_df)

    base_df = pred_df.merge(monthly_cash_df, on="user_id", how="left", validate="many_to_one")
    base_df["spent_mtd"] = base_df.apply(
        lambda row: lookup_spent_mtd(spent_lookup, row["user_id"], row["date"]), axis=1
    )
    base_df["future_available_7d"] = base_df.apply(
        lambda row: compute_future_available_7d(
            row["date"], float(row["monthly_available_cash"]), float(row["spent_mtd"])
        ),
        axis=1,
    )
    base_df["true_risk_ratio"] = base_df.apply(
        lambda row: compute_risk_ratio(float(row["y_true"]), float(row["future_available_7d"])),
        axis=1,
    )
    y_true = base_df["y_true"].to_numpy(dtype=float)
    return {
        "y_true": y_true,
        "future_available_7d": base_df["future_available_7d"].to_numpy(dtype=float),
        "true_alarm": base_df["true_risk_ratio"].apply(risk_ratio_to_alarm).to_numpy(dtype=int),
        "true_level": base_df["true_risk_ratio"].apply(risk_ratio_to_level).tolist(),
        "mae_norm": max(float(np.mean(np.abs(y_true))), 1.0),
        "rmse_norm": max(float(np.sqrt(np.mean(y_true ** 2))), 1.0),
    }


val_keep_mask = ~val_prediction_input_df["user_id"].astype(str).isin(SELECTION_EXCLUDE_USER_IDS)
val_keep_mask_np = val_keep_mask.to_numpy()
selection_val_prediction_input_df = val_prediction_input_df.loc[val_keep_mask].reset_index(drop=True)
selection_val_metric_context = build_metric_context(selection_val_prediction_input_df, split_metadata_df)
val_metric_context = build_metric_context(val_prediction_input_df, split_metadata_df)
train_metric_context = build_metric_context(train_prediction_input_df, split_metadata_df)


def risk_levels_to_int(levels: list[str]) -> np.ndarray:
    mapping = {"no_alarm": 0, "low_risk": 1, "mid_risk": 2, "high_risk": 3}
    return np.array([mapping[level] for level in levels], dtype=np.int64)


formal_train_labels = risk_levels_to_int(train_metric_context["true_level"])
formal_val_labels = risk_levels_to_int(val_metric_context["true_level"])
train_fav_7d = train_metric_context["future_available_7d"].reshape(-1, 1).astype(np.float32)
val_fav_7d = val_metric_context["future_available_7d"].reshape(-1, 1).astype(np.float32)
train_true_ratio = (
    train_metric_context["y_true"] / np.maximum(train_metric_context["future_available_7d"], 1e-8)
).reshape(-1, 1).astype(np.float32)
train_ordinal_targets = np.column_stack([
    (train_true_ratio.ravel() >= 0.8).astype(np.float32),
    (train_true_ratio.ravel() >= 1.0).astype(np.float32),
    (train_true_ratio.ravel() >= 1.2).astype(np.float32),
]).astype(np.float32)
train_boundary_weights = (
    1.0
    + 1.25 * np.exp(-np.abs(train_true_ratio.ravel() - 0.8) / 0.20)
    + 1.00 * np.exp(-np.abs(train_true_ratio.ravel() - 1.0) / 0.20)
    + 0.75 * np.exp(-np.abs(train_true_ratio.ravel() - 1.2) / 0.20)
).reshape(-1, 1).astype(np.float32)


def evaluate_raw_predictions(y_pred: np.ndarray, metric_context: dict) -> dict:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    fav_7d = metric_context["future_available_7d"]
    pred_ratio = np.array([compute_risk_ratio(float(p), float(f)) for p, f in zip(y_pred, fav_7d)])
    pred_alarm = np.array([risk_ratio_to_alarm(r) for r in pred_ratio], dtype=int)
    pred_level = [risk_ratio_to_level(r) for r in pred_ratio]

    reg = compute_regression_metrics(metric_context["y_true"], y_pred)
    bin_m = compute_binary_alarm_metrics(metric_context["true_alarm"], pred_alarm)
    cls_m = compute_4class_risk_metrics(metric_context["true_level"], pred_level)
    return {
        "MAE": reg["MAE"],
        "RMSE": reg["RMSE"],
        "Binary_F1": bin_m["F1-score"],
        "Weighted_F1": cls_m["Weighted F1"],
    }


def validation_selection_score(metrics: dict, metric_context: dict) -> float:
    return (
        0.72 * metrics["Binary_F1"]
        + 0.18 * metrics["Weighted_F1"]
        - 0.06 * (metrics["RMSE"] / metric_context["rmse_norm"])
        - 0.04 * (metrics["MAE"] / metric_context["mae_norm"])
    )


def compute_validation_metrics(model: nn.Module) -> dict:
    model.eval()
    with torch.no_grad():
        reg_out, _ = model(torch.tensor(X_val, dtype=torch.float32).to(device))
    raw_pred = target_scaler.inverse_transform(reg_out.cpu().numpy()[val_keep_mask_np]).ravel()
    return evaluate_raw_predictions(np.maximum(raw_pred, 0.0), selection_val_metric_context)

# ── Class weights ─────────────────────────────────────────────────────────────
label_counts  = Counter(train_labels.tolist())
total_samples = len(train_labels)
class_weights = torch.tensor(
    [total_samples / (NUM_CLASSES * Counter(formal_train_labels.tolist()).get(i, 1)) for i in range(NUM_CLASSES)],
    dtype=torch.float32
)
print(f"  Focal class weights: {class_weights.numpy().round(2)}")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ── 模型架構（Multi-task）─────────────────────────────────────────────────────
class GRUWithAttentionMT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, num_classes=4):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)   # 回歸頭
        self.cls_head   = nn.Linear(hidden_size // 2, num_classes)   # 分類頭
        self.relu       = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        hidden  = self.relu(self.fc1(out))
        return self.fc2(hidden), self.cls_head(hidden)


def load_pretrained_mt():
    """載入 pretrain 權重，新增分類頭（隨機初始化）"""
    ckpt  = torch.load(ARTIFACTS_DIR / "pretrain_aligned_gru.pth", map_location=device)
    model = GRUWithAttentionMT(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT, NUM_CLASSES)
    pretrained = ckpt["model_state"]
    current    = model.state_dict()
    for k, v in pretrained.items():
        if k in current:
            current[k] = v
    model.load_state_dict(current)
    return model


# ── Ensemble Finetune ─────────────────────────────────────────────────────────
print(f"\n🚀 Ensemble MT Finetune（seeds={SEEDS}，MT_ALPHA={MT_ALPHA}）...")

for seed in SEEDS:
    save_path = ARTIFACTS_DIR / f"finetune_aligned_gru_seed{seed}.pth"

    if os.path.exists(save_path):
        print(f"\n  Seed {seed}：已存在，跳過")
        continue

    print(f"\n{'='*55}")
    print(f"  Seed {seed}")
    print(f"{'='*55}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model      = load_pretrained_mt().to(device)
    l1_crit    = nn.HuberLoss(delta=HUBER_DELTA)
    mse_crit   = nn.MSELoss()
    ce_crit    = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights.to(device))

    encoder_params = (list(model.gru.parameters()) +
                      list(model.attention.parameters()) +
                      list(model.layer_norm.parameters()))
    head_params    = (list(model.fc1.parameters()) +
                      list(model.fc2.parameters()) +
                      list(model.cls_head.parameters()))

    # Phase 1：freeze encoder，只訓練 heads
    for p in encoder_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)
    phase2_started = False

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
            torch.tensor(formal_train_labels, dtype=torch.long),
            torch.tensor(y_train_raw.astype(np.float32)),
            torch.tensor(train_fav_7d),
            torch.tensor(train_true_ratio),
            torch.tensor(train_ordinal_targets),
            torch.tensor(train_boundary_weights),
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_val),
            torch.tensor(formal_val_labels, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )

    best_val_loss    = float("inf")
    best_val_score   = -float("inf")
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Phase 2：unfreeze encoder，切換為 differential LR optimizer
        if epoch == FREEZE_EPOCHS + 1 and not phase2_started:
            for p in encoder_params:
                p.requires_grad = True
            current_head_lr = optimizer.param_groups[0]["lr"]
            optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": ENCODER_LR},
                {"params": head_params,    "lr": current_head_lr},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)
            phase2_started = True
            print(f"  🔓 Phase 2 開始：unfreeze encoder（encoder_lr={ENCODER_LR}, head_lr={current_head_lr:.2e}）")

        model.train()
        epoch_reg = epoch_cls = 0.0

        for X_b, y_b, lbl_b, y_raw_b, fav_b, ratio_b, ord_b, weight_b in train_loader:
            X_b, y_b, lbl_b = X_b.to(device), y_b.to(device), lbl_b.to(device)
            y_raw_b = y_raw_b.to(device)
            fav_b = fav_b.to(device)
            ratio_b = ratio_b.to(device)
            ord_b = ord_b.to(device)
            weight_b = weight_b.to(device)
            optimizer.zero_grad()
            reg_out, cls_out = model(X_b)
            r_loss = l1_crit(reg_out, y_b) + MSE_WEIGHT * mse_crit(reg_out, y_b)
            c_loss = ce_crit(cls_out, lbl_b)
            raw_pred = reg_out * float(target_scaler.scale_[0]) + float(target_scaler.mean_[0])
            pred_ratio = raw_pred / torch.clamp(fav_b, min=1e-6)
            ratio_loss = F.smooth_l1_loss(pred_ratio * weight_b, ratio_b * weight_b, beta=0.20)
            ordinal_logits = torch.cat([
                (pred_ratio - 0.8) / BOUNDARY_TEMP,
                (pred_ratio - 1.0) / BOUNDARY_TEMP,
                (pred_ratio - 1.2) / BOUNDARY_TEMP,
            ], dim=1)
            ordinal_loss = F.binary_cross_entropy_with_logits(
                ordinal_logits,
                ord_b,
                weight=weight_b.expand_as(ord_b),
            )
            loss = (
                r_loss
                + MT_ALPHA * c_loss
                + RISK_RATIO_ALPHA * ratio_loss
                + ORDINAL_ALPHA * ordinal_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_reg += r_loss.item()
            epoch_cls   += c_loss.item()

        epoch_reg /= len(train_loader)
        epoch_cls   /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b, lbl_b in val_loader:
                reg_out, _ = model(X_b.to(device))
                y_b = y_b.to(device)
                val_loss += (l1_crit(reg_out, y_b) + MSE_WEIGHT * mse_crit(reg_out, y_b)).item()
        val_loss /= len(val_loader)
        val_metrics = compute_validation_metrics(model)
        val_score = validation_selection_score(val_metrics, selection_val_metric_context)

        scheduler.step(val_loss)
        if epoch % 10 == 0:
            head_lr = optimizer.param_groups[-1]["lr"]
            print(
                f"  Epoch {epoch:3d}  Reg: {epoch_reg:.4f}  CE: {epoch_cls:.4f}  "
                f"ValLoss: {val_loss:.6f}  ValScore: {val_score:.6f}  LR: {head_lr:.6f}"
            )

        if (val_score > best_val_score + 1e-8) or (
            abs(val_score - best_val_score) <= 1e-8 and val_loss < best_val_loss
        ):
            best_val_loss    = val_loss
            best_val_score   = val_score
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : best_val_loss,
                "val_score"  : best_val_score,
                "val_metrics": best_val_metrics,
                "seed"       : seed,
                "version"    : "ibm_finetune_mt_val_metric_selected",
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹️  Early stopping")
                break

    if best_val_metrics:
        print(
            f"  Seed {seed} 最佳 Val Score: {best_val_score:.6f} "
            f"MAE={best_val_metrics['MAE']:.2f} RMSE={best_val_metrics['RMSE']:.2f} "
            f"Binary_F1={best_val_metrics['Binary_F1']:.4f} "
            f"Weighted_F1={best_val_metrics['Weighted_F1']:.4f}"
        )
    else:
        print(f"  Seed {seed} 最佳 Val Loss: {best_val_loss:.6f}")

print(f"\n🎉 MT Finetune 完成！→ 下一步：執行 5_predict_aligned.py")
