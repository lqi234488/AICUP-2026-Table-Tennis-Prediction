import pandas as pd
import sys
import os
import glob

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

def get_latest_submission(folder_path="all_submissions"):
    """
    從指定資料夾中尋找最新的 submission_*.csv 檔案
    """
    if not os.path.exists(folder_path):
        return None
    
    # 尋找所有符合 pattern 的檔案
    search_path = os.path.join(folder_path, "submission_*.csv")
    files = glob.glob(search_path)
    
    if not files:
        return None
    
    # 根據檔案修改時間排序，取最後一個
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def check_submission(file_path=None):
    # 如果沒指定檔案，就去 all_submissions 找最新的
    if file_path is None:
        latest = get_latest_submission("all_submissions")
        if latest:
            file_path = latest
            print(f"🕒 自動偵測到最新檔案：{file_path}")
        else:
            file_path = 'submission.csv' # 沒找到就 fallback 到原本的預設
            print(f"💡 找不到 all_submissions 中的檔案，使用預設：{file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 錯誤：找不到檔案 '{file_path}'")
        return

    try:
        df = pd.read_csv(file_path)
        total_rows = len(df)
        
        print(f"\n" + "═"*60)
        print(f"📊 Submission 直觀檢查報告: {file_path}")
        print("═"*60)
        print(f"🔹 總預測筆數：{total_rows} 筆")

        # 1. pointId (球落點) 分佈檢查
        print(f"\n📍 [pointId 球落點分佈]")
        print("-" * 60)
        ptr_counts = df['pointId'].value_counts().sort_index()
        
        for val, count in ptr_counts.items():
            name = POINT_MAP.get(int(val), "未知/其他")
            percentage = (count / total_rows) * 100
            bar = "█" * int(percentage / 2)
            print(f"ID {int(val):2d} | {name:<12} | {count:6d} 筆 ({percentage:5.1f}%) {bar}")
            
        if len(ptr_counts) <= 1:
            # 取得目前的預測值名稱
            pred_val = int(ptr_counts.index[0])
            name = POINT_MAP.get(pred_val, f"ID {pred_val}")
            print(f"\n🚨 嚴重警告：pointId 發生坍塌！目前全部預測為：{name}")

        # 2. actionId (擊球方式) 分佈檢查
        print(f"\n🏓 [actionId 擊球方式分佈 - 前 15 名]")
        print("-" * 60)
        act_counts = df['actionId'].value_counts().head(15)
        
        for val, count in act_counts.items():
            name = ACTION_MAP.get(int(val), "未知/其他")
            percentage = (count / total_rows) * 100
            bar = "░" * int(percentage / 2)
            print(f"ID {int(val):2d} | {name:<8} | {count:6d} 筆 ({percentage:5.1f}%) {bar}")

        # 3. serverGetPoint 分佈
        if 'serverGetPoint' in df.columns:
            print(f"\n🏆 [serverGetPoint 得分預測]")
            sgp_counts = df['serverGetPoint'].value_counts()
            for val, count in sgp_counts.items():
                label = "得分" if int(val) == 1 else "未得分"
                print(f"{label}: {count:6d} 筆 ({count/total_rows*100:5.1f}%)")

        print("═"*60 + "\n")

    except Exception as e:
        print(f"❌ 發生異常：{str(e)}")

if __name__ == "__main__":
    # 如果命令列有給參數，就用參數；否則就傳 None 啟動自動搜尋
    target = sys.argv[1] if len(sys.argv) > 1 else None
    check_submission(target)