"""
Step 4：BiGRU Aligned Finetune (方案 2：兩階段訓練終極版)
==============================
策略：
1. 兩階段訓練：Phase 1 凍結 Encoder (護腦) -> Phase 2 全局解凍微調 (適應極端值)
2. Loss 調整：L1 + 0.1 * MSE，取得 MAE 與 RMSE 的最佳平衡
3. 搭配 ReduceLROnPlateau 進行精細收斂
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── 1. 路徑鎖定 ──────────────────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ML_UTILS_DIR = MY_DIR.parents[1] / "ml"

sys.path.insert(0, str(MY_DIR))
sys.path.insert(0, str(ML_UTILS_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention
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

# ── 2. 超參數設定 ────────────────────────────────────────────────────────
INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1
BATCH_SIZE = 32

# 【終極修改】兩階段訓練參數
PHASE1_EPOCHS = 20        # Phase 1: 凍結特徵層，只練 Head
PHASE2_EPOCHS = 25        # Phase 2: 全局解凍，微調特徵
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5          # 極小學習率，保護 Pretrain 知識

PATIENCE = 8
WEIGHT_DECAY = 1e-5
MSE_WEIGHT = 0.1          # 調低 MSE 權重，確保 MAE 不會被反噬
SENSITIVITY_EXCLUDE_USER_IDS = ["user14"]

# 保持 30 個 Seed 確保檢定公平
SEEDS = [
    42, 123, 777, 456, 789, 999, 2024, 0, 7, 13, 21, 100, 314, 1234, 9999,
    11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666
]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"⚙️  使用設備: {device}")

# ── 3. 載入資料 ──────────────────────────────────────────────────────────
print(f"📂 載入個人資料...")
X_train = np.load(ARTIFACTS_DIR / "personal_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "personal_y_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "personal_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "personal_y_val.npy")
val_user_ids = np.load(ARTIFACTS_DIR / "personal_val_user_ids.npy")

with open(ARTIFACTS_DIR / "personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
y_val_raw = target_scaler.inverse_transform(y_val)
metadata_df = pd.read_csv(ARTIFACTS_DIR / "metadata.csv")
split_metadata_df = metadata_df[["user_id", "date", "split"]]
val_meta = metadata_df[metadata_df["split"] == "val"].reset_index(drop=True)
val_prediction_input_df = pd.DataFrame({
    "user_id": val_meta["user_id"],
    "date": val_meta["date"],
    "y_true": y_val_raw.ravel(),
    "y_pred": y_val_raw.ravel(),
})
val_keep_mask = ~val_prediction_input_df["user_id"].astype(str).isin(SENSITIVITY_EXCLUDE_USER_IDS)
val_keep_mask_np = val_keep_mask.to_numpy()
selection_val_input_df = val_prediction_input_df.loc[val_keep_mask].reset_index(drop=True)

PRETRAIN_WEIGHT_PATH = ARTIFACTS_DIR / "pretrain_bigru.pth"

# ── 4. 載入預訓練模型 ──────────────────────────────────────────────────────────
def load_pretrained():
    if not PRETRAIN_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到預訓練大腦: {PRETRAIN_WEIGHT_PATH}")
    ckpt = torch.load(PRETRAIN_WEIGHT_PATH, map_location=device, weights_only=True)
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model

def build_metric_context(prediction_input_df: pd.DataFrame, split_metadata_df: pd.DataFrame) -> dict:
    pred_df = _prepare_prediction_input(prediction_input_df)
    split_df = _prepare_split_metadata(split_metadata_df)
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
    return {
        "y_true": base_df["y_true"].to_numpy(dtype=float),
        "future_available_7d": base_df["future_available_7d"].to_numpy(dtype=float),
        "true_alarm": base_df["true_risk_ratio"].apply(risk_ratio_to_alarm).to_numpy(dtype=int),
        "true_level": base_df["true_risk_ratio"].apply(risk_ratio_to_level).tolist(),
        "mae_norm": max(float(np.mean(np.abs(base_df["y_true"].to_numpy(dtype=float)))), 1.0),
        "rmse_norm": max(float(np.sqrt(np.mean(base_df["y_true"].to_numpy(dtype=float) ** 2))), 1.0),
    }

selection_val_context = build_metric_context(selection_val_input_df, split_metadata_df)

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
        preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
    raw_pred = target_scaler.inverse_transform(preds[val_keep_mask_np]).ravel()
    return evaluate_raw_predictions(np.maximum(raw_pred, 0.0), selection_val_context)

# ── 5. 訓練迴圈 ──────────────────────────────────────────────────────────
print(f"\n🚀 開始微調 (方案 2：兩階段解凍 + Blended Loss)...")

for seed in SEEDS:
    save_path = ARTIFACTS_DIR / f"finetune_bigru_seed{seed}.pth"
    if save_path.exists():
        print(f"⏩ Seed {seed} 已存在，跳過")
        continue

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_pretrained().to(device)
    
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

    best_val_loss = float("inf")
    best_val_score = -float("inf")
    best_val_metrics = None

    # ==========================================
    # 🛑 Phase 1: 凍結 Encoder，只訓練 Head
    # ==========================================
    for name, param in model.named_parameters():
        if "fc" not in name:  
            param.requires_grad = False
            
    optimizer_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

    print(f"🔥 Seed {seed} [Phase 1: 凍結]...", end=" ")
    for epoch in range(1, PHASE1_EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer_p1.zero_grad()
            preds = model(X_b)
            loss = criterion_l1(preds, y_b) + MSE_WEIGHT * criterion_mse(preds, y_b)
            loss.backward()
            optimizer_p1.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)
                preds_v = model(X_v)
                v_loss += (criterion_l1(preds_v, y_v) + MSE_WEIGHT * criterion_mse(preds_v, y_v)).item()
        v_loss /= len(val_loader)

        val_metrics = compute_validation_metrics(model)
        val_score = validation_selection_score(val_metrics, selection_val_context)
        if (val_score > best_val_score + 1e-8) or (
            abs(val_score - best_val_score) <= 1e-8 and v_loss < best_val_loss
        ):
            best_val_loss = v_loss
            best_val_score = val_score
            best_val_metrics = val_metrics
            torch.save({"model_state": model.state_dict()}, save_path)

    # ==========================================
    # 🟢 Phase 2: 全局解凍，極小學習率微調
    # ==========================================
    # 解凍所有層
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode='min', factor=0.5, patience=5)
    
    print(f"➡️ [Phase 2: 解凍]...", end=" ")
    patience_counter = 0
    
    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer_p2.zero_grad()
            preds = model(X_b)
            loss = criterion_l1(preds, y_b) + MSE_WEIGHT * criterion_mse(preds, y_b)
            loss.backward()
            optimizer_p2.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                X_v, y_v = X_v.to(device), y_v.to(device)
                preds_v = model(X_v)
                v_loss += (criterion_l1(preds_v, y_v) + MSE_WEIGHT * criterion_mse(preds_v, y_v)).item()
        v_loss /= len(val_loader)
        
        scheduler.step(v_loss)

        val_metrics = compute_validation_metrics(model)
        val_score = validation_selection_score(val_metrics, selection_val_context)

        if (val_score > best_val_score + 1e-8) or (
            abs(val_score - best_val_score) <= 1e-8 and v_loss < best_val_loss
        ):
            best_val_loss = v_loss
            best_val_score = val_score
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save({"model_state": model.state_dict()}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: 
                break
    
    if best_val_metrics is None:
        print(f"完成！最佳 Val Loss: {best_val_loss:.6f}")
    else:
        print(
            f"完成！最佳 Val Score: {best_val_score:.6f} "
            f"Val MAE: {best_val_metrics['MAE']:.2f} "
            f"Val RMSE: {best_val_metrics['RMSE']:.2f} "
            f"Val Binary_F1: {best_val_metrics['Binary_F1']:.4f} "
            f"Val Weighted_F1: {best_val_metrics['Weighted_F1']:.4f}"
        )

print("\n🎉 兩階段微調結束！現在可以跑 python 5_predict_bigru.py 了！")
