import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def get_latest_submission(folder_path="all_submissions"):
    files = glob.glob(os.path.join(folder_path, "submission_*.csv"))
    if not files: return 'submission.csv'
    return max(files, key=os.path.getmtime)

def local_validation(test_file='test.csv', sub_file=None):
    if sub_file is None:
        sub_file = get_latest_submission()
    
    if not os.path.exists(test_file) or not os.path.exists(sub_file):
        print(f"❌ 錯誤：找不到檔案。請確認 {test_file} 與 {sub_file} 存在。")
        return

    # 1. 讀取資料
    test_df = pd.read_csv(test_file)
    sub_df = pd.read_csv(sub_file)

    # 2. 提取 test.csv 中每個 Rally 的最後一拍 (這才是真正的 Target)
    # 根據你的 build_sequences 邏輯，測試集只預測最後一拍
    test_targets = test_df.sort_values(['rally_uid', 'strikeNumber']).groupby('rally_uid').tail(1)
    
    # 3. 合併預測值與真實值
    # 我們用 rally_uid 當 Key，把真實答案 (gt_) 跟預測答案 (pred_) 併在一起
    res = pd.merge(
        test_targets[['rally_uid', 'actionId', 'pointId', 'serverGetPoint']],
        sub_df[['rally_uid', 'actionId', 'pointId', 'serverGetPoint']],
        on='rally_uid',
        suffixes=('_gt', '_pred')
    )

    print(f"\n" + "═"*60)
    print(f"🔍 本地對答案報告 (Validation Result)")
    print(f"📄 預測檔案：{sub_file}")
    print(f"📄 參考答案：{test_file}")
    print(f"🔹 比對筆數：{len(res)} 筆")
    print("═"*60)

    # 4. 檢查 ID 範圍 (診斷 0.13 分的關鍵！)
    print(f"\n🔢 [ID 範圍診斷]")
    for col in ['actionId', 'pointId']:
        gt_min, gt_max = res[f'{col}_gt'].min(), res[f'{col}_gt'].max()
        pred_min, pred_max = res[f'{col}_pred'].min(), res[f'{col}_pred'].max()
        print(f"{col:8} | 真實範圍: {gt_min:2d}~{gt_max:2d} | 預測範圍: {pred_min:2d}~{pred_max:2d}")
        if pred_min < gt_min or pred_max > gt_max:
            print(f"   ⚠️ 警告：預測 ID 範圍與真實 ID 不符！這就是導致低分的原因。")

    # 5. 計算各項指標
    print(f"\n🏆 [各項指標得分]")
    print("-" * 60)
    
    # Action F1 (Macro)
    f1_action = f1_score(res['actionId_gt'], res['actionId_pred'], average='macro')
    acc_action = accuracy_score(res['actionId_gt'], res['actionId_pred'])
    print(f"🏓 擊球方式 (Action) | F1: {f1_action:.4f} | Acc: {acc_action:.4f}")

    # Point F1 (Macro)
    f1_point = f1_score(res['pointId_gt'], res['pointId_pred'], average='macro')
    acc_point = accuracy_score(res['pointId_gt'], res['pointId_pred'])
    print(f"📍 球落點 (Point)   | F1: {f1_point:.4f} | Acc: {acc_point:.4f}")

    # SGP AUC (或 Acc)
    try:
        auc_sgp = roc_auc_score(res['serverGetPoint_gt'], res['serverGetPoint_pred'])
        print(f"🏆 得分預測 (SGP)    | AUC: {auc_sgp:.4f}")
    except:
        acc_sgp = accuracy_score(res['serverGetPoint_gt'], (res['serverGetPoint_pred'] > 0.5).astype(int))
        print(f"🏆 得分預測 (SGP)    | Acc: {acc_sgp:.4f} (無法計算 AUC，改顯示 Acc)")

    # 綜合得分 (自定義)
    overall = (f1_action * 0.4) + (f1_point * 0.4) + (0.2 * (auc_sgp if 'auc_sgp' in locals() else acc_sgp))
    print("-" * 60)
    print(f"⭐ 綜合預估得分：{overall:.4f}")

    # 6. 列出前 5 筆錯誤案例分析
    print(f"\n❌ [錯誤案例抽樣 - Action]")
    errors = res[res['actionId_gt'] != res['actionId_pred']].head(5)
    if not errors.empty:
        for _, row in errors.iterrows():
            print(f"UID: {row['rally_uid']:4.0f} | 真實: {int(row['actionId_gt']):2d} | 預測: {int(row['actionId_pred']):2d}")

    print("═"*60 + "\n")

if __name__ == "__main__":
    local_validation()