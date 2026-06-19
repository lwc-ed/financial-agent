"""
[NO-ALIGN pipeline] Step 4：個人資料微調（兩階段、30 seeds）
==========================================================
忠實複製原 4_finetune_bigru.py 的兩階段訓練 + Blended Loss + 驗證選模邏輯，
僅：(1) 特徵維度來自 raw、(2) 讀寫 ml_temp/artifacts_noalign、
    (3) 重用原本的 model_bigru 與 ml_walmart/output_eval_utils（只讀 import）。

輸出 → ml_temp/artifacts_noalign/finetune_bigru_seed{seed}.pth
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent / "ml_ibm" / "bigru_TL_alignment"
EVAL = HERE.parent / "ml_walmart"
for p in (HERE, ORIG, EVAL):
    sys.path.insert(0, str(p))

from model_bigru import BiGRUWithAttention  # noqa: E402
from no_alignment_utils import RAW_FEATURE_COLS  # noqa: E402
from output_eval_utils import (  # noqa: E402  （只讀重用評估工具）
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

ART = HERE / "artifacts_noalign"

INPUT_SIZE = len(RAW_FEATURE_COLS)  # 10
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1
BATCH_SIZE = 32

PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 25
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5
PATIENCE = 8
WEIGHT_DECAY = 1e-5
MSE_WEIGHT = 0.1
SENSITIVITY_EXCLUDE_USER_IDS = ["user14"]

SEEDS = [
    42, 123, 777, 456, 789, 999, 2024, 0, 7, 13, 21, 100, 314, 1234, 9999,
    11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666,
]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️  設備: {device}")

X_train = np.load(ART / "personal_X_train.npy")
y_train = np.load(ART / "personal_y_train.npy")
X_val = np.load(ART / "personal_X_val.npy")
y_val = np.load(ART / "personal_y_val.npy")

with open(ART / "personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
y_val_raw = target_scaler.inverse_transform(y_val)
metadata_df = pd.read_csv(ART / "metadata.csv")
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

PRETRAIN_WEIGHT_PATH = ART / "pretrain_bigru.pth"


def load_pretrained():
    if not PRETRAIN_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到預訓練權重: {PRETRAIN_WEIGHT_PATH}")
    ckpt = torch.load(PRETRAIN_WEIGHT_PATH, map_location=device, weights_only=True)
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model


def build_metric_context(prediction_input_df, split_metadata_df):
    pred_df = _prepare_prediction_input(prediction_input_df)
    split_df = _prepare_split_metadata(split_metadata_df)
    raw_txn_df = _prepare_transactions(None)
    monthly_cash_df = compute_monthly_available_cash(raw_txn_df, split_df)
    spent_lookup = build_spent_mtd_lookup(raw_txn_df)

    base_df = pred_df.merge(monthly_cash_df, on="user_id", how="left", validate="many_to_one")
    base_df["spent_mtd"] = base_df.apply(
        lambda r: lookup_spent_mtd(spent_lookup, r["user_id"], r["date"]), axis=1)
    base_df["future_available_7d"] = base_df.apply(
        lambda r: compute_future_available_7d(
            r["date"], float(r["monthly_available_cash"]), float(r["spent_mtd"])), axis=1)
    base_df["true_risk_ratio"] = base_df.apply(
        lambda r: compute_risk_ratio(float(r["y_true"]), float(r["future_available_7d"])), axis=1)
    return {
        "y_true": base_df["y_true"].to_numpy(dtype=float),
        "future_available_7d": base_df["future_available_7d"].to_numpy(dtype=float),
        "true_alarm": base_df["true_risk_ratio"].apply(risk_ratio_to_alarm).to_numpy(dtype=int),
        "true_level": base_df["true_risk_ratio"].apply(risk_ratio_to_level).tolist(),
        "mae_norm": max(float(np.mean(np.abs(base_df["y_true"].to_numpy(dtype=float)))), 1.0),
        "rmse_norm": max(float(np.sqrt(np.mean(base_df["y_true"].to_numpy(dtype=float) ** 2))), 1.0),
    }


selection_val_context = build_metric_context(selection_val_input_df, split_metadata_df)


def evaluate_raw_predictions(y_pred, ctx):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    fav = ctx["future_available_7d"]
    pred_ratio = np.array([compute_risk_ratio(float(p), float(f)) for p, f in zip(y_pred, fav)])
    pred_alarm = np.array([risk_ratio_to_alarm(r) for r in pred_ratio], dtype=int)
    pred_level = [risk_ratio_to_level(r) for r in pred_ratio]
    reg = compute_regression_metrics(ctx["y_true"], y_pred)
    bin_m = compute_binary_alarm_metrics(ctx["true_alarm"], pred_alarm)
    cls_m = compute_4class_risk_metrics(ctx["true_level"], pred_level)
    return {"MAE": reg["MAE"], "RMSE": reg["RMSE"],
            "Binary_F1": bin_m["F1-score"], "Weighted_F1": cls_m["Weighted F1"]}


def validation_selection_score(m, ctx):
    return (0.72 * m["Binary_F1"] + 0.18 * m["Weighted_F1"]
            - 0.06 * (m["RMSE"] / ctx["rmse_norm"]) - 0.04 * (m["MAE"] / ctx["mae_norm"]))


def compute_validation_metrics(model):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
    raw_pred = target_scaler.inverse_transform(preds[val_keep_mask_np]).ravel()
    return evaluate_raw_predictions(np.maximum(raw_pred, 0.0), selection_val_context)


print("\n🚀 開始微調（兩階段解凍 + Blended Loss）...")
for seed in SEEDS:
    save_path = ART / f"finetune_bigru_seed{seed}.pth"
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

    # Phase 1：凍結 Encoder，只訓練 Head
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    optimizer_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)

    print(f"🔥 Seed {seed} [Phase 1]...", end=" ")
    for _ in range(1, PHASE1_EPOCHS + 1):
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
                pv = model(X_v)
                v_loss += (criterion_l1(pv, y_v) + MSE_WEIGHT * criterion_mse(pv, y_v)).item()
        v_loss /= len(val_loader)
        vm = compute_validation_metrics(model)
        vs = validation_selection_score(vm, selection_val_context)
        if (vs > best_val_score + 1e-8) or (abs(vs - best_val_score) <= 1e-8 and v_loss < best_val_loss):
            best_val_loss, best_val_score, best_val_metrics = v_loss, vs, vm
            torch.save({"model_state": model.state_dict()}, save_path)

    # Phase 2：全局解凍，極小學習率微調
    for param in model.parameters():
        param.requires_grad = True
    optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode="min", factor=0.5, patience=5)

    print("➡️ [Phase 2]...", end=" ")
    patience_counter = 0
    for _ in range(1, PHASE2_EPOCHS + 1):
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
                pv = model(X_v)
                v_loss += (criterion_l1(pv, y_v) + MSE_WEIGHT * criterion_mse(pv, y_v)).item()
        v_loss /= len(val_loader)
        scheduler.step(v_loss)
        vm = compute_validation_metrics(model)
        vs = validation_selection_score(vm, selection_val_context)
        if (vs > best_val_score + 1e-8) or (abs(vs - best_val_score) <= 1e-8 and v_loss < best_val_loss):
            best_val_loss, best_val_score, best_val_metrics = v_loss, vs, vm
            patience_counter = 0
            torch.save({"model_state": model.state_dict()}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_val_metrics is None:
        print(f"完成！Val Loss: {best_val_loss:.6f}")
    else:
        print(f"完成！Val MAE: {best_val_metrics['MAE']:.2f} "
              f"Binary_F1: {best_val_metrics['Binary_F1']:.4f} "
              f"Weighted_F1: {best_val_metrics['Weighted_F1']:.4f}")

print("\n🎉 兩階段微調結束！接著跑 python 5_predict.py")
