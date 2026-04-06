# 2026 AI CUP: 基於時序資料之桌球戰術與結果預測 🏓

本專案旨在參與 **2026 AI CUP** 競賽，利用深度學習技術分析桌球比賽中的時序資料。
核心任務在於根據一回合中前 $N-1$ 拍的擊球資訊，精準預測第 $N$ 拍的戰術動作、落點位置及最終得分結果。
https://www.aidea-web.tw/topic/3f9662e8-9d18-4c6a-9332-103eded3a399

## 🚀 專案目標
* **時序建模**：處理變長序列的擊球資料。
* **多任務學習 (Multi-task Learning)**：同時預測三項關鍵指標：
  1. `actionId`：擊球動作類型。
  2. `pointId`：球的落點區域 (1-10 區)。
  3. `serverGetPoint`：發球方是否得分。

## 🛠 技術棧
* **語言**：Python 3.12+
* **深度學習框架**：TensorFlow / Keras 3 (TensorFlow Backend)
* **模型架構**：多輸出 LSTM (Long Short-Term Memory)
* **資料處理**：Pandas, NumPy, Scikit-learn

## 📊 資料集說明
資料來源為競賽提供之 CSV 檔案，包含以下核心特徵：
- `rally_uid`: 回合唯一識別碼。
- `strikeNumber`: 擊球序號。
- `spinId`: 球的旋轉類型。
- `positionId`: 擊球者的位置。
- `scoreSelf` / `scoreOther`: 當前比分。

## 🧠 模型設計邏輯
本專案採用 ** (Multi-output Model)** 架構：
1. **滑動窗口處理**：將原始資料轉換為固定長度的時序序列（預設 $MAX\_LEN = 10$）。
2. **共享層**：使用雙層 LSTM 提取高維度的戰術特徵。
3. **獨立輸出頭**：針對不同的預測目標，分設三個輸出端（Softmax 與 Sigmoid），並給予不同的損失權重 (Loss Weights) 以優化落點預測精度。

## 📈 目前進度
- [x] 資料初步探索 (EDA) 與清洗。
- [x] 實作滑動窗口資料生成器。
- [x] 建立多輸出 LSTM Baseline 模型。
- [ ] 特徵工程優化（剔除性別、賽次等雜訊特徵）。
- [ ] 調整超參數以提升 Validation Accuracy。
