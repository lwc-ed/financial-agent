"""
gru_hgbr/extract_embeddings.py
================================
從 fine-tuned GRU（V4）提取 context embedding。

做法：
  - 載入 ml_gru/artificats/finetune_gru_v4.pth（已 fine-tune 到個人資料的最佳模型）
  - 對每個 30 天視窗，取 GRU + Attention 輸出的 context vector（64 維）
  - 這個向量捕捉了序列的時序特徵（趨勢、週期性、動量）
  - 後續由 HGBR 負責最終預測，避免 GRU 在小資料上 overfit

為什麼用 V4 fine-tuned 而非 pretrain？
  - fine-tuned V4 已適應個人消費資料的分佈
  - 提取的 embedding 對個人財務任務更相關
  - 比直接用 Walmart pretrain 的 embedding 更有意義
"""

import numpy as np
import torch
import torch.nn as nn
import pickle

ML_GRU_ARTIFACTS  = "../ml_gru/artificats"
GRU_MODEL_FILE    = "finetune_gru.pth"   # artifacts 裡實際的檔名
ARTIFACTS_DIR    = "artifacts"


class GRUModel(nn.Module):
    """與 ml_gru finetune_gru.pth 實際架構一致：GRU + 單一 FC，無 Attention"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])   # 取最後一個時間步

    def get_embedding(self, x):
        """取最後時間步的 hidden state 作為 embedding（64 維）"""
        gru_out, _ = self.gru(x)
        return gru_out[:, -1, :]   # shape: (B, hidden_size)


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

# ─────────────────────────────────────────
# 載入 fine-tuned GRU V4
# ─────────────────────────────────────────
print(f"\n📦 載入 {GRU_MODEL_FILE}...")
ckpt = torch.load(f"{ML_GRU_ARTIFACTS}/{GRU_MODEL_FILE}", map_location=device)
hp   = ckpt["hyperparams"]

model = GRUModel(
    hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"  ✅ 載入完成（hidden={hp['hidden_size']}, layers={hp['num_layers']}）")
print(f"     Embedding 維度：{hp['hidden_size']}  (= context vector size)")

# ─────────────────────────────────────────
# 載入序列資料
# ─────────────────────────────────────────
print("\n📂 載入 GRU 序列資料...")
gru_X_train = np.load(f"{ARTIFACTS_DIR}/gru_X_train.npy")
gru_X_val   = np.load(f"{ARTIFACTS_DIR}/gru_X_val.npy")
gru_X_test  = np.load(f"{ARTIFACTS_DIR}/gru_X_test.npy")
print(f"  Train: {gru_X_train.shape}  Val: {gru_X_val.shape}  Test: {gru_X_test.shape}")

# ─────────────────────────────────────────
# 提取 embedding（batch 處理避免 OOM）
# ─────────────────────────────────────────
def extract_embeddings(X, batch_size=256):
    """逐批提取 embedding，回傳 numpy array (N, hidden_size)"""
    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            emb   = model.get_embedding(batch).cpu().numpy()
            all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)

print("\n🔍 提取 embedding...")
emb_train = extract_embeddings(gru_X_train)
emb_val   = extract_embeddings(gru_X_val)
emb_test  = extract_embeddings(gru_X_test)

print(f"  Train embedding: {emb_train.shape}")
print(f"  Val   embedding: {emb_val.shape}")
print(f"  Test  embedding: {emb_test.shape}")

# ─────────────────────────────────────────
# 與 HGBR 扁平特徵拼接
# embedding (64-dim) + hgbr_features (22-dim) = 86-dim
# ─────────────────────────────────────────
print("\n🔗 拼接 GRU embedding + HGBR 扁平特徵...")
hgbr_X_train = np.load(f"{ARTIFACTS_DIR}/hgbr_X_train.npy")
hgbr_X_val   = np.load(f"{ARTIFACTS_DIR}/hgbr_X_val.npy")
hgbr_X_test  = np.load(f"{ARTIFACTS_DIR}/hgbr_X_test.npy")

combined_train = np.concatenate([emb_train, hgbr_X_train], axis=1).astype(np.float32)
combined_val   = np.concatenate([emb_val,   hgbr_X_val],   axis=1).astype(np.float32)
combined_test  = np.concatenate([emb_test,  hgbr_X_test],  axis=1).astype(np.float32)

print(f"  Combined Train: {combined_train.shape}  (embedding {emb_train.shape[1]} + flat {hgbr_X_train.shape[1]})")
print(f"  Combined Val:   {combined_val.shape}")
print(f"  Combined Test:  {combined_test.shape}")

# ─────────────────────────────────────────
# 儲存
# ─────────────────────────────────────────
np.save(f"{ARTIFACTS_DIR}/combined_X_train.npy", combined_train)
np.save(f"{ARTIFACTS_DIR}/combined_X_val.npy",   combined_val)
np.save(f"{ARTIFACTS_DIR}/combined_X_test.npy",  combined_test)
np.save(f"{ARTIFACTS_DIR}/emb_train.npy",         emb_train)
np.save(f"{ARTIFACTS_DIR}/emb_val.npy",           emb_val)
np.save(f"{ARTIFACTS_DIR}/emb_test.npy",          emb_test)

print("\n✅ 完成！下一步：python train_hgbr.py")
