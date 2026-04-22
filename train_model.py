"""
桌球擊球預測 - PyTorch 多任務學習
=====================================
Task 1: actionId   (19 classes) -> Macro F1  (weight 0.4)
Task 2: pointId    (10 classes) -> Macro F1  (weight 0.4)
Task 3: serverGetPoint (binary) -> AUC-ROC   (weight 0.2)

資料結構：每個 rally_uid 有多拍序列，用前 n-1 拍預測第 n 拍
# 定義類別對照表
POINT_MAP = {
    0: "無/不計", # 修正：ID 0 通常代表沒落點（失誤）
    1: "正手位置短球", 2: "中間短球", 3: "反手位置短球",
    4: "正手位置半出台球", 5: "中路半出台球", 6: "反手位置短半出台球",
    7: "正手位置長球", 8: "中間長球", 9: "反手位置長球"
}

ACTION_MAP = {
    0: "無", 1: "拉球", 2: "反拉", 3: "殺球", 4: "擰球",
    5: "快帶", 6: "推擠", 7: "挑撥", 8: "拱球", 9: "磕球",
    10: "搓球", 11: "擺短", 12: "削球", 13: "擋球", 14: "放高球",
    15: "傳統", 16: "勾手", 17: "逆旋轉", 18: "下蹲式"
}
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
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'  如果沒有超過五張會變成CPU 不如不指定讓他自動抓
import datetime
import collections
torch.autograd.set_detect_anomaly(True) 

# ──────────────────────────────────────────────
# 參數設定
# ──────────────────────────────────────────────
SEED = 77
MAX_SEQ_LEN = 15
BATCH_SIZE = 256
EPOCHS = 30  #50
N_FOLDS = 3  #5
PATIENCE = 40           
LR = 5e-5               
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.2           
LOSS_W = {'action': 0.4, 'point': 1.5, 'sgp': 0.2}
N_ACTION_CLASSES = 19
N_POINT_CLASSES = 10
DEBUG_MODE = False
OVERFIT_TEST = False

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 2. 特徵欄位定義
# ──────────────────────────────────────────────
# 衍生特徵
train_df['score_diff'] = train_df['scoreSelf'] - train_df['scoreOther']
test_df['score_diff']  = test_df['scoreSelf']  - test_df['scoreOther']

train_df['is_first_strike'] = (train_df['strikeNumber'] == 1).astype(int)
test_df['is_first_strike']  = (test_df['strikeNumber']  == 1).astype(int)

train_df['is_early_rally'] = (train_df['strikeNumber'] <= 3).astype(int)
test_df['is_early_rally']  = (test_df['strikeNumber']  <= 3).astype(int)
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
NUM_COLS = ['strikeNumber', 'scoreSelf', 'scoreOther',
            'score_diff', 'is_first_strike', 'is_early_rally']

# ──────────────────────────────────────────────
# 3. Encoding
# ──────────────────────────────────────────────
all_df = pd.concat([train_df, test_df], ignore_index=True)

# 建立一個包含所有需要 Encode 的欄位清單
ALL_ENCODE_COLS = CAT_COLS + ['pointId']

encoders = {}
for col in ALL_ENCODE_COLS:
    le = LabelEncoder()
    # 確保用數值排序，這樣 ID 0~18 才會對應到索引 0~18
    all_df[col] = le.fit_transform(all_df[col].fillna(-1).astype(int)) 
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
            if len(history) == 0:
                cat_seq = (target_row[CAT_COLS].values.astype(np.int64) + 1).reshape(1, -1)
                num_seq = target_row[NUM_COLS].values.astype(np.float32).reshape(1, -1)
            else:
                cat_seq = (history[CAT_COLS].values.astype(np.int64)) + 1
                num_seq = history[NUM_COLS].values.astype(np.float32)
            
            if len(cat_seq) > MAX_SEQ_LEN:
                cat_seq = cat_seq[-MAX_SEQ_LEN:]
                num_seq = num_seq[-MAX_SEQ_LEN:]

            sequences.append({
                'rally_uid': rally_uid,
                'cat_seq': cat_seq, 'num_seq': num_seq, 'seq_len': len(cat_seq),
                'actionId': 0,
                'pointId': 0,
                'serverGetPoint': 0,
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
test_seqs  = build_sequences(test_df,  is_test=True)

if DEBUG_MODE:
    #檢查前 5 個序列的目標值
    print("\n [數據抽檢] 檢查 sequences 裡的目標值：")
    for i in range(5):
        s = train_seqs[i]
        print(f"Rally: {s['rally_uid']} | actionId: {s['actionId']} | pointId: {s['pointId']}")

    #檢查 pointId 的數值範圍
    all_p_targets = [s['pointId'] for s in train_seqs]
    print(f"\n pointId 目標值範圍: {min(all_p_targets)} ~ {max(all_p_targets)}")
    #統計所有訓練序列中的 pointId 數量
    p_counts = collections.Counter([s['pointId'] for s in train_seqs])

    print("\n [全量標籤統計] pointId 在訓練集中的分佈：")
    total = sum(p_counts.values())
    for pid, count in sorted(p_counts.items()):
        percentage = (count / total) * 100
        print(f"ID {pid}: {count:6d} 筆 ({percentage:5.1f}%) {'█' * int(percentage/2)}")
    print(f" pointId 不重複數值數量: {len(set(all_p_targets))}") 

    lengths = pd.Series([s['seq_len'] for s in train_seqs])
    print("\n📊 序列長度分佈：")
    print(lengths.describe().round(2))
    print(f"\n分位數：")
    print(lengths.quantile([0.5, 0.75, 0.9, 0.95]).round(0))
    zero_len = [s for s in test_seqs if s['seq_len'] == 0]
    print(f"DEBUG: seq_len=0 的測試序列數量 = {len(zero_len)}")

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

        action_seq = [-1] * (max_len - 1) + [s['actionId']]
        point_seq  = [-1] * (max_len - 1) + [s['pointId']]

        cat_list.append(cat_pad)
        num_list.append(num_pad)
        mask_list.append(mask)
        action_list.append(action_seq)
        point_list.append(point_seq)
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

tr_seqs_subset = [train_seqs[i] for i in tr_idx]
val_seqs_subset = [train_seqs[i] for i in val_idx]

def print_dist(seqs, name):
    action_counts = collections.Counter([s['actionId'] for s in seqs])
    point_counts  = collections.Counter([s['pointId']  for s in seqs])
    total = len(seqs)
    print(f"\n{'='*20} {name} (共 {total} 筆) {'='*20}")
    print("actionId 分佈（前5常見）:")
    for k, v in sorted(action_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  class {k:2d}: {v:5d} ({v/total*100:.1f}%)")
    print("pointId 分佈（前5常見）:")
    for k, v in sorted(point_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  class {k:2d}: {v:5d} ({v/total*100:.1f}%)")
if DEBUG_MODE:
    print_dist(tr_seqs_subset,  "Train")
    print_dist(val_seqs_subset, "Val")

if OVERFIT_TEST:
    tiny_seqs = train_seqs[:100]
    tr_set  = RallyDataset(tiny_seqs)
    val_set = RallyDataset(tiny_seqs)  # 刻意用同一份，看能不能記住
    print("⚠️  過擬合測試模式：只用 100 筆資料")

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
            nn.Linear(D_MODEL // 2, N_ACTION_CLASSES),
        )
        self.head_point = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, N_POINT_CLASSES),
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
        positions = positions.clamp(max=MAX_SEQ_LEN)
        x = x + self.pos_enc(positions)

        x = self.transformer(x, src_key_padding_mask=mask)

        out_action = self.head_action(x)   # (B, L, 19)
        out_point  = self.head_point(x)    # (B, L, 10)

        real_mask_float = (~mask).unsqueeze(-1).float()   # (B, L, 1)
        last_idx = (real_mask_float.squeeze(-1).cumsum(dim=1) - 1).argmax(dim=1)  # (B,)
        last_idx = last_idx.clamp(min=0, max=x.shape[1] - 1)
        last_idx_exp = last_idx.view(B, 1, 1).expand(B, 1, D_MODEL)
        h_last = x.gather(1, last_idx_exp).squeeze(1)  # (B, D_MODEL)
        h_last = torch.nan_to_num(h_last, nan=0.0)
        out_sgp = self.head_sgp(h_last).squeeze(-1)

        return out_action, out_point, out_sgp

model = MultiTaskTransformer().to(DEVICE)

# ──────────────────────────────────────────────
# 7. 損失函數 & 優化器
# ──────────────────────────────────────────────
def compute_class_weights(seqs, key, n_classes):
    labels = np.array([s[key] for s in seqs])
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    
    # 關鍵修正：限制權重最大值，避免罕見類別梯度爆炸
    weights = np.clip(weights, a_min=None, a_max=10.0) 
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

action_weights = compute_class_weights(train_seqs, 'actionId', 19)
point_weights  = compute_class_weights(train_seqs, 'pointId',  10)

# 關鍵修正：移除 label_smoothing，避免與 class weights 發生數學衝突
criterion_action = nn.CrossEntropyLoss(weight=action_weights, ignore_index=-1)
criterion_point  = nn.CrossEntropyLoss(weight=point_weights,  ignore_index=-1)
criterion_sgp    = nn.BCEWithLogitsLoss()

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

    with torch.set_grad_enabled(training):
        for batch in loader:
            cat  = batch['cat'].to(DEVICE)
            num  = batch['num'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            y_action = batch['action'].to(DEVICE)
            y_point  = batch['point'].to(DEVICE)
            y_sgp    = batch['sgp'].to(DEVICE)

            logit_a, logit_p, logit_s = model(cat, num, mask)

            loss_a = criterion_action(logit_a.view(-1, N_ACTION_CLASSES), y_action.view(-1))
            loss_p = criterion_point( logit_p.view(-1, N_POINT_CLASSES), y_point.view(-1))
            loss_s = criterion_sgp(logit_s, y_sgp)
            loss   = (LOSS_W['action'] * loss_a
                    + LOSS_W['point']  * loss_p
                    + LOSS_W['sgp']    * loss_s)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()

            total_loss += loss.item() * len(y_action)

            a_flat   = y_action.view(-1).cpu().numpy()
            a_pred   = logit_a.view(-1, 19).argmax(dim=1).cpu().numpy()
            mask_a   = (a_flat != -1)
            all_action_pred.extend(a_pred[mask_a])
            all_action_true.extend(a_flat[mask_a])
            p_flat   = y_point.view(-1).cpu().numpy()
            p_pred   = logit_p.view(-1, 10).argmax(dim=1).cpu().numpy()
            mask_p   = (p_flat != -1)
            all_point_pred.extend(p_pred[mask_p])
            all_point_true.extend(p_flat[mask_p])
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
# 9. K-Fold 訓練迴圈
# ──────────────────────────────────────────────
gkf = GroupKFold(n_splits=N_FOLDS)
rally_uids = np.array([s['rally_uid'] for s in train_seqs])

all_fold_preds = []  # 存每個 fold 的測試集預測
all_history = []
overall_best_score = 0.0

print("\n=== K-Fold Training Start ===")

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_seqs, groups=rally_uids)):
    print(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")

    tr_set  = RallyDataset([train_seqs[i] for i in tr_idx])
    val_set = RallyDataset([train_seqs[i] for i in val_idx])
    tr_loader  = DataLoader(tr_set,  batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 每個 fold 重新初始化模型和優化器
    model = MultiTaskTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )

    best_score = 0.0
    best_state = None
    epochs_no_improve = 0
    fold_history = []

    for epoch in range(1, EPOCHS + 1):
        tr_metrics  = run_epoch(tr_loader,  training=True)
        val_metrics = run_epoch(val_loader, training=False)
        scheduler.step(val_metrics['overall'])

        fold_history.append({
            'fold': fold + 1,
            'epoch': epoch,
            **{f'tr_{k}': v for k, v in tr_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"Fold {fold+1} | Epoch {epoch:03d}/{EPOCHS} | "
                  f"Loss {tr_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                  f"F1-action {tr_metrics['f1_action']:.4f}/{val_metrics['f1_action']:.4f} | "
                  f"F1-point {tr_metrics['f1_point']:.4f}/{val_metrics['f1_point']:.4f} | "
                  f"AUC {tr_metrics['auc_sgp']:.4f}/{val_metrics['auc_sgp']:.4f} | "
                  f"Overall {val_metrics['overall']:.4f}")

        if val_metrics['overall'] > best_score:
            if not np.isnan(val_metrics['loss']):
                best_score = val_metrics['overall']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                
                # 存這個 fold 的最佳模型
                fold_model_path = os.path.join(OUTPUT_DIR, f"best_model_fold{fold+1}.pt")
                torch.save(best_state, fold_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"Fold {fold+1} best val score: {best_score:.4f}")
    all_history.extend(fold_history)

    if best_score > overall_best_score:
        overall_best_score = best_score

    # ── 用這個 fold 的最佳模型預測測試集 ──
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()

    fold_action, fold_point, fold_sgp, fold_uids = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(te_loader):
            cat  = batch['cat'].to(DEVICE)
            num  = batch['num'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)

            logit_a, logit_p, logit_s = model(cat, num, mask)

            if i == 0 and DEBUG_MODE and fold == 0:
                print(f"DEBUG: logit_p[0, -1, :] = {logit_p[0, -1, :].cpu().numpy()}")

            last_action = logit_a[:, -1, :]
            last_point  = logit_p[:, -1, :]

            # 存 softmax 機率，之後 ensemble 時取平均
            fold_action.append(torch.softmax(last_action, dim=1).cpu().numpy())
            fold_point.append(torch.softmax(last_point,   dim=1).cpu().numpy())
            fold_sgp.append(torch.sigmoid(logit_s).cpu().numpy())
            fold_uids.extend(batch['uid'])

    all_fold_preds.append({
        'action': np.concatenate(fold_action, axis=0),  # (N_test, N_ACTION_CLASSES)
        'point':  np.concatenate(fold_point,  axis=0),  # (N_test, N_POINT_CLASSES)
        'sgp':    np.concatenate(fold_sgp,    axis=0),  # (N_test,)
    })

print(f"\nOverall best val score across folds: {overall_best_score:.4f}")
pd.DataFrame(all_history).to_csv("training_history.csv", index=False)
print("Saved training_history.csv")

# ──────────────────────────────────────────────
# 10. Ensemble 預測 & 輸出
# ──────────────────────────────────────────────

# 所有 fold 的預測取平均
avg_action = np.mean([p['action'] for p in all_fold_preds], axis=0)  # (N_test, 19)
avg_point  = np.mean([p['point']  for p in all_fold_preds], axis=0)  # (N_test, 10)
avg_sgp    = np.mean([p['sgp']    for p in all_fold_preds], axis=0)  # (N_test,)

all_action_pred = avg_action.argmax(axis=1).tolist()
all_point_pred  = avg_point.argmax(axis=1).tolist()
all_sgp_prob    = avg_sgp.tolist()

# pointId inverse transform
le_point = encoders['pointId']
final_point_ids = le_point.inverse_transform(np.array(all_point_pred))

if DEBUG_MODE:
    print(f"DEBUG: all_point_pred 前20筆 = {all_point_pred[:20]}")
    print(f"DEBUG: all_point_pred 唯一值 = {set(all_point_pred)}")
    print(f"DEBUG: 轉換後的前 5 筆 = {final_point_ids[:5]}")

submission = pd.DataFrame({
    'rally_uid':      fold_uids,
    'actionId':       all_action_pred,
    'pointId':        final_point_ids,
    'serverGetPoint': all_sgp_prob,
})

# NaN 檢查
nan_count = submission['serverGetPoint'].isnull().sum()
if nan_count > 0:
    print(f"⚠️ 警告：偵測到 {nan_count} 筆預測值為 NaN，已自動填充為 0")
    submission['serverGetPoint'] = submission['serverGetPoint'].fillna(0)

# serverGetPoint 四捨五入
submission['serverGetPoint'] = submission['serverGetPoint'].round().astype(int)

# 輸出檔案
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
filename = os.path.join(OUTPUT_DIR, f"submission_{timestamp}.csv")
submission.to_csv(filename, index=False)
print(f"✅ 檔案已成功產出：{filename}")
