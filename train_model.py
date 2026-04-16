"""
桌球擊球預測 - PyTorch 多任務學習
=====================================
Task 1: actionId   (19 classes) -> Macro F1  (weight 0.4)
Task 2: pointId    (10 classes) -> Macro F1  (weight 0.4)
Task 3: serverGetPoint (binary) -> AUC-ROC   (weight 0.2)

資料結構：每個 rally_uid 有多拍序列，用前 n-1 拍預測第 n 拍
"""

"""
桌球擊球預測 - PyTorch 多任務學習 (修正版)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'  如果沒有超過五張會變成CPU 不如不指定讓他自動抓
import datetime
import collections
torch.autograd.set_detect_anomaly(True) # 加在程式最開頭

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
SEED = 77
MAX_SEQ_LEN = 30
BATCH_SIZE = 256
EPOCHS = 1000
PATIENCE = 25           # Early Stopping 容忍 Epoch 數
LR = 5e-5               # 調降初始學習率，避免振盪
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.2           # 提高 Dropout 增加正則化強度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "all_submissions"

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────
# 1. 讀資料
# ──────────────────────────────────────────────
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# ──────────────────────────────────────────────
# 2. 特徵欄位定義
# ──────────────────────────────────────────────
FEATURE_COLS = [
    'sex', 'match', 'numberGame', 'rally_id',
    'strikeNumber', 'scoreSelf', 'scoreOther',
    'gamePlayerId', 'gamePlayerOtherId',
    'strikeId', 'handId', 'strengthId', 'spinId',
    'positionId'
]

TARGET_COLS = ['actionId', 'pointId', 'serverGetPoint']

CAT_COLS = [
    'sex', 'strikeId', 'handId', 'strengthId', 'spinId', 
    'positionId', 'gamePlayerId', 'gamePlayerOtherId'  
]
NUM_COLS = ['strikeNumber', 'scoreSelf', 'scoreOther']

# ──────────────────────────────────────────────
# 3. Encoding
# ──────────────────────────────────────────────
all_df = pd.concat([train_df, test_df], ignore_index=True)

# 建立一個包含所有需要 Encode 的欄位清單
ALL_ENCODE_COLS = CAT_COLS + ['pointId']

encoders = {}
for col in ALL_ENCODE_COLS:
    le = LabelEncoder()
    all_df[col] = le.fit_transform(all_df[col].astype(str)) 
    encoders[col] = le

n_train = len(train_df)
train_df = all_df.iloc[:n_train].copy()
test_df  = all_df.iloc[n_train:].copy()

CAT_VOCAB = {col: all_df[col].max() + 2 for col in CAT_COLS}

num_mean = train_df[NUM_COLS].mean()
num_std  = train_df[NUM_COLS].std().replace(0, 1)

train_df[NUM_COLS] = (train_df[NUM_COLS] - num_mean) / num_std
test_df[NUM_COLS]  = (test_df[NUM_COLS]  - num_mean) / num_std

# ──────────────────────────────────────────────
# 4. 序列建構
# ──────────────────────────────────────────────
def build_sequences(df, is_test=False):
    sequences = []
    groups = df.sort_values(['rally_uid', 'strikeNumber']).groupby('rally_uid')

    for rally_uid, g in groups:
        g = g.reset_index(drop=True)

        if is_test:
            # 測試集維持不變：用前面所有拍數預測最後一拍
            history = g.iloc[:-1]
            target_row = g.iloc[-1]
            
            cat_seq = (history[CAT_COLS].values.astype(np.int64)) + 1
            num_seq = history[NUM_COLS].values.astype(np.float32)
            
            sequences.append({
                'rally_uid': rally_uid,
                'cat_seq': cat_seq, 'num_seq': num_seq, 'seq_len': len(history),
                'actionId': int(target_row['actionId']),
                'pointId': int(target_row['pointId']),
                'serverGetPoint': int(target_row['serverGetPoint']),
            })
        else:
            # 訓練集：使用滑動視窗！
            # 每一拍都可以作為 Target（從第 2 拍開始，因為前面要有 history）
            for i in range(1, len(g)):
                history = g.iloc[:i]    # 這一拍之前的全部是 history
                target_row = g.iloc[i]  # 當前這一拍是我們要學的答案
                
                cat_seq = (history[CAT_COLS].values.astype(np.int64)) + 1
                num_seq = history[NUM_COLS].values.astype(np.float32)

                # 限制最大序列長度
                if len(cat_seq) > MAX_SEQ_LEN:
                    cat_seq = cat_seq[-MAX_SEQ_LEN:]
                    num_seq = num_seq[-MAX_SEQ_LEN:]

                sequences.append({
                    'rally_uid': rally_uid,
                    'cat_seq': cat_seq, 'num_seq': num_seq, 'seq_len': len(cat_seq),
                    'actionId': int(target_row['actionId']),
                    'pointId': int(target_row['pointId']),
                    'serverGetPoint': int(target_row['serverGetPoint']),
                })

    return sequences

print("Building sequences...")
train_seqs = build_sequences(train_df, is_test=False)
# 檢查前 5 個序列的目標值
#print("\n [數據抽檢] 檢查 sequences 裡的目標值：")
#for i in range(5):
#    s = train_seqs[i]
#    print(f"Rally: {s['rally_uid']} | actionId: {s['actionId']} | pointId: {s['pointId']}")

# 檢查 pointId 的數值範圍
#all_p_targets = [s['pointId'] for s in train_seqs]
#print(f"\n pointId 目標值範圍: {min(all_p_targets)} ~ {max(all_p_targets)}")
# 統計所有訓練序列中的 pointId 數量
#p_counts = collections.Counter([s['pointId'] for s in train_seqs])

#print("\n [全量標籤統計] pointId 在訓練集中的分佈：")
#total = sum(p_counts.values())
#for pid, count in sorted(p_counts.items()):
#    percentage = (count / total) * 100
#    print(f"ID {pid}: {count:6d} 筆 ({percentage:5.1f}%) {'█' * int(percentage/2)}")
#print(f" pointId 不重複數值數量: {len(set(all_p_targets))}")
###
test_seqs  = build_sequences(test_df,  is_test=True)
print(f"  Train sequences: {len(train_seqs)}")
print(f"  Test  sequences: {len(test_seqs)}")

# ──────────────────────────────────────────────
# 5. Dataset & DataLoader
# ──────────────────────────────────────────────
def collate_fn(batch):
    max_len = max(s['seq_len'] for s in batch)

    cat_list, num_list, mask_list = [], [], []
    action_list, point_list, sgp_list, uid_list = [], [], [], []

    for s in batch:
        L = s['seq_len']
        pad = max_len - L

        cat_pad = np.pad(s['cat_seq'], ((pad, 0), (0, 0)), mode='constant', constant_values=0)
        num_pad = np.pad(s['num_seq'], ((pad, 0), (0, 0)), mode='constant', constant_values=0.0)
        mask    = [True] * pad + [False] * L

        cat_list.append(cat_pad)
        num_list.append(num_pad)
        mask_list.append(mask)
        action_list.append(s['actionId'])
        point_list.append(s['pointId'])
        sgp_list.append(s['serverGetPoint'])
        uid_list.append(s['rally_uid'])

    return {
        'cat':    torch.tensor(np.array(cat_list),  dtype=torch.long),
        'num':    torch.tensor(np.array(num_list),  dtype=torch.float),
        'mask':   torch.tensor(mask_list,           dtype=torch.bool),
        'action': torch.tensor(action_list,         dtype=torch.long),
        'point':  torch.tensor(point_list,          dtype=torch.long),
        'sgp':    torch.tensor(sgp_list,            dtype=torch.float),
        'uid':    uid_list,
    }

class RallyDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return self.seqs[idx]

rally_uids = np.array([s['rally_uid'] for s in train_seqs])
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr_idx, val_idx = next(gss.split(train_seqs, groups=rally_uids))

tr_set  = RallyDataset([train_seqs[i] for i in tr_idx])
val_set = RallyDataset([train_seqs[i] for i in val_idx])
te_set  = RallyDataset(test_seqs)

tr_loader  = DataLoader(tr_set,  batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
te_loader  = DataLoader(te_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

# ──────────────────────────────────────────────
# 6. 模型：Transformer Encoder + 多任務輸出頭
# ──────────────────────────────────────────────
class MultiTaskTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(CAT_VOCAB[col], min(32, CAT_VOCAB[col] // 2 + 2), padding_idx=0)
            for col in CAT_COLS
        })
        embed_dim = sum(min(32, CAT_VOCAB[col] // 2 + 2) for col in CAT_COLS)
        num_dim   = len(NUM_COLS)
        input_dim = embed_dim + num_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, D_MODEL),
            nn.LayerNorm(D_MODEL),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        self.pos_enc = nn.Embedding(MAX_SEQ_LEN + 1, D_MODEL, padding_idx=0)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        self.head_action = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 19),
        )
        self.head_point = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 10),
        )
        self.head_sgp = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def forward(self, cat, num, mask):
        B, L, _ = cat.shape
        emb_parts = [self.embeddings[col](cat[:, :, i]) for i, col in enumerate(CAT_COLS)]
        x = torch.cat(emb_parts + [num], dim=-1)

        x = self.input_proj(x)

        positions = torch.arange(1, L + 1, device=x.device).unsqueeze(0)
        positions = positions.masked_fill(mask, 0)
        x = x + self.pos_enc(positions)

        x = self.transformer(x, src_key_padding_mask=mask)

        real_mask = ~mask
        last_idx = real_mask.long().cumsum(dim=1).argmax(dim=1)
        last_idx_exp = last_idx.view(B, 1, 1).expand(B, 1, D_MODEL)
        h = x.gather(1, last_idx_exp).squeeze(1)

        out_action = self.head_action(h)
        out_point  = self.head_point(h)
        out_sgp    = self.head_sgp(h).squeeze(-1)

        return out_action, out_point, out_sgp

model = MultiTaskTransformer().to(DEVICE)

# ──────────────────────────────────────────────
# 7. 損失函數 & 優化器
# ──────────────────────────────────────────────
def compute_class_weights(seqs, key, n_classes):
    labels = np.array([s[key] for s in train_seqs])
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    
    # 關鍵修正：限制權重最大值，避免罕見類別梯度爆炸
    weights = np.clip(weights, a_min=None, a_max=10.0) 
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

action_weights = compute_class_weights(train_seqs, 'actionId', 19)
point_weights  = compute_class_weights(train_seqs, 'pointId',  10)

# 關鍵修正：移除 label_smoothing，避免與 class weights 發生數學衝突
criterion_action = nn.CrossEntropyLoss(weight=action_weights)
criterion_point  = nn.CrossEntropyLoss(weight=point_weights)
criterion_sgp    = nn.BCEWithLogitsLoss()

LOSS_W = {'action': 0.4, 'point': 1.5, 'sgp': 0.2}

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

# 改變 Scheduler 設定，改用 ReduceLROnPlateau 以根據 Validation Loss 動態調整學習率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
)

# ──────────────────────────────────────────────
# 8. 訓練 & 驗證函數
# ──────────────────────────────────────────────
def run_epoch(loader, training=True):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_action_pred, all_action_true = [], []
    all_point_pred,  all_point_true  = [], []
    all_sgp_prob,    all_sgp_true    = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            cat  = batch['cat'].to(DEVICE)
            num  = batch['num'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            y_action = batch['action'].to(DEVICE)
            y_point  = batch['point'].to(DEVICE)
            y_sgp    = batch['sgp'].to(DEVICE)

            logit_a, logit_p, logit_s = model(cat, num, mask)

            loss_a = criterion_action(logit_a, y_action)
            loss_p = criterion_point(logit_p,  y_point)
            loss_s = criterion_sgp(logit_s,    y_sgp)
            loss   = (LOSS_W['action'] * loss_a
                    + LOSS_W['point']  * loss_p
                    + LOSS_W['sgp']    * loss_s)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()

            total_loss += loss.item() * len(y_action)

            all_action_pred.extend(logit_a.argmax(dim=1).cpu().numpy())
            all_action_true.extend(y_action.cpu().numpy())
            all_point_pred.extend(logit_p.argmax(dim=1).cpu().numpy())
            all_point_true.extend(y_point.cpu().numpy())
            all_sgp_prob.extend(torch.sigmoid(logit_s).detach().cpu().numpy())
            all_sgp_true.extend(y_sgp.cpu().numpy())
    
    avg_loss = total_loss / len(all_action_true)

    # 修正：移除 labels=[0, 1] 確保計算所有類別的 Macro F1
    f1_action = f1_score(all_action_true, all_action_pred, average='macro', zero_division=0)
    f1_point  = f1_score(all_point_true,  all_point_pred,  average='macro', zero_division=0)
    
    try:
        auc_sgp = roc_auc_score(all_sgp_true, all_sgp_prob)
    except ValueError:
        auc_sgp = 0.5

    overall = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc_sgp

    return {
        'loss':      avg_loss,
        'f1_action': f1_action,
        'f1_point':  f1_point,
        'auc_sgp':   auc_sgp,
        'overall':   overall,
    }

# ──────────────────────────────────────────────
# 9. 訓練迴圈
# ──────────────────────────────────────────────
best_score = 0.0
best_state = None
history = []
epochs_no_improve = 0  # 實作 Early Stopping 計數器

print("\n=== Training Start ===")
for epoch in range(1, EPOCHS + 1):
    tr_metrics  = run_epoch(tr_loader,  training=True)
    val_metrics = run_epoch(val_loader, training=False)
    
    # Scheduler 根據 Validation Overall Score 調整 LR
    scheduler.step(val_metrics['overall'])

    history.append({'epoch': epoch, **{f'tr_{k}': v for k, v in tr_metrics.items()},
                                     **{f'val_{k}': v for k, v in val_metrics.items()}})

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"Loss {tr_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
              f"F1-action {tr_metrics['f1_action']:.4f}/{val_metrics['f1_action']:.4f} | "
              f"F1-point {tr_metrics['f1_point']:.4f}/{val_metrics['f1_point']:.4f} | "
              f"AUC {tr_metrics['auc_sgp']:.4f}/{val_metrics['auc_sgp']:.4f} | "
              f"Overall {val_metrics['overall']:.4f}")

    if val_metrics['overall'] > best_score:
        if not np.isnan(val_metrics['loss']):
            best_score = val_metrics['overall']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # 1. 強制覆蓋 logic
            model_path = 'best_model.pt'
            if os.path.exists(model_path):
                try: os.remove(model_path)
                except: pass
            
            # 2. 存入硬碟 (確保就算當機也有一份在硬碟裡)
            torch.save(best_state, model_path)
            
            # 3. 備份一份帶時間和分數的 (方便 check2.py 追蹤)
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            # 這裡修正了變數名稱，改用 val_metrics['loss']
            ver_path = os.path.join(OUTPUT_DIR, f"model_loss_{val_metrics['loss']:.4f}_{timestamp}.pt")
            torch.save(best_state, ver_path)
            
            print(f"  ★ 發現更強模型！Score: {best_score:.4f} | 已存檔：{ver_path}")
            epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # 觸發 Early Stopping
    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch}")
        break

print(f"\nBest validation overall score: {best_score:.4f}")
pd.DataFrame(history).to_csv("training_history.csv", index=False)
print("Saved training_history.csv")

# ──────────────────────────────────────────────
# 10. 預測 & 11. 輸出 & 12. 儲存模型
# ──────────────────────────────────────────────
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
model.eval()

all_uids, all_action_pred, all_point_pred, all_sgp_prob = [], [], [], []

with torch.no_grad():
    for i, batch in enumerate(te_loader):
        cat  = batch['cat'].to(DEVICE)
        num  = batch['num'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)

        logit_a, logit_p, logit_s = model(cat, num, mask)

        # --- 檢查開始 ---
        if i == 0: # 只看第一個 Batch 即可
            print(f"DEBUG: logit_p Shape = {logit_p.shape}") 
            # 預期輸出應該是 [256, 10] (假設 Batch 是 256, 類別是 10)
            
            # 看看前 5 筆資料的預測索引
            preds = logit_p.argmax(dim=1).cpu().numpy()
            print(f"DEBUG: 前 5 筆預測索引 = {preds[:5]}")
            
            # 看看原始 Logits 的數值（檢查是否數值都一模一樣導致坍塌）
            print(f"DEBUG: 第一筆資料的 Logits = {logit_p[0].cpu().numpy()}")
        # --- 檢查結束 ---
        all_uids.extend(batch['uid'])
        all_action_pred.extend(logit_a.argmax(dim=1).cpu().numpy())
        all_point_pred.extend(logit_p.argmax(dim=1).cpu().numpy())
        all_sgp_prob.extend(torch.sigmoid(logit_s).cpu().numpy())

submission = pd.DataFrame({
    'rally_uid':      all_uids,
    'actionId':       all_action_pred,
    'pointId':        all_point_pred,
    'serverGetPoint': all_sgp_prob, # 1. 先存原始機率
})

# 2. 檢查並填充 NaN (這很重要！)
nan_count = submission['serverGetPoint'].isnull().sum()
if nan_count > 0:
    print(f"⚠️ 警告：偵測到 {nan_count} 筆預測值為 NaN，已自動填充為 0")
    submission['serverGetPoint'] = submission['serverGetPoint'].fillna(0)

# 3. 執行四捨五入並轉成整數
submission['serverGetPoint'] = submission['serverGetPoint'].round().astype(int)
# 1. 取得 pointId 的 LabelEncoder
le_point = encoders['pointId']

# 2. 獲取模型產出的原始索引 (0~9)
# 假設 all_point_pred 是 logit_p.argmax(dim=1) 的集合
raw_indices = np.array(all_point_pred)

# 3. 檢查：如果你在訓練時用了 +1，那麼 argmax 得到的 0 其實對應的是 encoders 裡的 class 0
# 但因為你儲存的是類別索引而非原始 ID，我們需要把它換回來
# 這裡要很小心，因為 LabelEncoder 不知道你手動加了 1
final_point_ids = le_point.inverse_transform(raw_indices)

# 更新 submission
submission['pointId'] = final_point_ids

print(f"DEBUG: 轉換後的前 5 筆 = {final_point_ids[:5]}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
filename = os.path.join(OUTPUT_DIR, f"submission_{timestamp}.csv")
model_filename = os.path.join(OUTPUT_DIR, f"best_model_{timestamp}.pt")
history_filename = os.path.join(OUTPUT_DIR, f"history_{timestamp}.csv")
submission.to_csv(filename, index=False)

# 強制覆蓋邏輯
model_path = 'best_model.pt'
if os.path.exists(model_path):
    try:
        os.remove(model_path) # 先手動砍掉，防止權限鎖定不讓蓋
    except:
        pass 

# 儲存最優權重
torch.save(best_state, model_path)

# 💡 建議：同時存一個帶時間的版本在 all_submissions，這才是真的保險
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
ver_path = os.path.join(OUTPUT_DIR, f"model_loss_{val_metrics['loss']:.4f}_{timestamp}.pt")
torch.save(best_state, ver_path)
print(f"✅ 檔案已成功產出：{filename}")

#print("\n=== Done ===")
#print(f"Final best validation score: {best_score:.4f}")
#print("  Files produced:")
#print("    submission.csv       <- 上傳用")
#print("    best_model.pt        <- 模型權重")
#print("    training_history.csv <- 訓練曲線")
