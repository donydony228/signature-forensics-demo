# 簽名比對輔助 Demo

一個**示範用**的離線簽名比對網站，目的是協助文書鑑識人員檢視「待鑑簽名和已知樣本差在哪裡」，並把每一個數字都攤開到具體的、看得懂的特徵上。

> ## ⚠️ 這是示範，不是鑑定工具
>
> 本專案的數值是用公開研究資料集（CEDAR，西方拉丁字母簽名）校準出來的，**不可用於實際案件**。
> 它沒有經過任何鑑識驗證程序，錯誤率也未在與案件相符的資料上量測過。
> 任何真偽判斷都應由合格的文書鑑識人員做出。

**English summary**: A demonstration web tool for offline handwritten-signature comparison, aimed at forensic document examiners. It extracts interpretable handcrafted features (geometric, stroke-direction, HOG, LBP, and others), expresses every difference in units of normal writer variation (σ), and decomposes the resulting score back into named features and an annotated heat map. Calibrated on the CEDAR dataset for demonstration only — **not validated for casework**.

## 設計理念

一般的簽名驗證系統用 CNN 抽特徵，準確率高，但沒辦法說明「為什麼判斷是這樣」。鑑識場合的需求剛好相反：**可解釋性優先於準確率**，因為結論要寫進報告、要能在法庭上被質問與討論。

所以這個專案刻意不用神經網路，改用人設計的特徵。每一項差異都能對應到鑑識人員本來就在用的語言——比例、重心、走向、提筆次數、筆畫長度、筆畫方向。

代價很明確：準確率比 CNN 差。在 CEDAR 上，手工特徵的等錯誤率是 15.1%，同一份切分下 SigNet 的深度特徵是 6.7%。這個取捨是刻意的。

## 現況

| 版本 | 連接埠 | 說明 |
|---|---|---|
| **教學版**（`app.py`） | 8000 | 說明整個流程：特徵 → 偏離幾個 σ → 距離門檻 → 二元判定 |
| **鑑識版**（`forensic_app.py`） | 8010 | 差值向量 + Elastic Net 線性模型，輸出證據強度（likelihood ratio），不下真偽結論 |

鑑識版的完整設計見 **[docs/SPEC.md](docs/SPEC.md)**。兩個版本的主要差異：

- 不給「本人／非本人」的結論，改報 likelihood ratio 加文字級距，由鑑識人員自行判斷
- 權重由資料學出來，而非所有特徵等權重
- 呈現的是「權重 × 差異」的證據分解，而不只是「差多少」
- 可匯出單一 HTML 報告，附進案件卷宗

## 方法概述

```
兩張簽名圖
  → 前處理：裁到筆跡範圍、等比縮放、Otsu 二值化
  → 8 族特徵共 3334 維（幾何、方向分布、小波、Radon、GLCM、HOG、LBP、SIFT）
  → 每一維換算成「偏離幾個 σ」
  → 教學版：距離門檻 ｜ 鑑識版：Elastic Net → 邏輯迴歸校準 → likelihood ratio
  → 拆回具名特徵與熱區圖
```

特徵族與維度：

| 族 | 維度 | 內容 |
|---|---|---|
| 幾何 | 7 | 寬高比、墨跡密度、質心、整體傾角、筆畫段數、筆畫總長÷高度 |
| 方向分布 | 10 | 骨架切線方向 8 箱 + 端點數、交叉點數 |
| 小波 | 10 | 3 層 db2 分解，各子帶能量佔比 |
| Radon | 36 | 每 5 度一個角度的投影能量佔比 |
| GLCM | 48 | 6 個統計量 × 2 距離 × 4 角度 |
| HOG | 2970 | 15×22 格 × 9 方向 |
| LBP | 250 | 5×5 格 × 10 種模式 |
| SIFT | 3 | 配對率、平均描述子距離、仿射 RANSAC 一致性（成對計算）|

σ 是「一個人重複簽自己名字時，這個特徵正常會變動多少」。把單位不同的特徵（寬高比約 2、墨跡密度約 0.05）換算成 σ 之後才能互相比較。

### 教學版目前的數字

CEDAR 前 30 人、每人 3 張參考簽名：

| 方法 | EER（全體共用閾值） | EER（每人各自閾值） | 誤收別人真簽名 |
|---|---|---|---|
| 只用幾何特徵 | 23.2% | 20.6% | 1.8% |
| 只用 HOG | 36.8% | 17.9% | 17.5% |
| 只用 LBP | 34.9% | 18.0% | 6.2% |
| 手工特徵合併 | 22.9% | 15.1% | 1.3% |
| SigNet 深度特徵（對照） | 16.9% | 6.7% | 1.8% |

EER 是誤拒率等於誤收率時的錯誤率，愈低愈好。**這些數字只在 CEDAR 上成立**，且閾值與測試用的是同一批資料，會偏樂觀。

## 安裝與執行

```bash
# 1. 環境
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt

# 2. 取得 sigver（前處理與 SigNet 對照用）
git clone --depth 1 https://github.com/luizgh/sigver.git

# 3. 下載 CEDAR 資料集（約 254MB）並解壓到 data/
mkdir -p data && cd data
curl -LO https://cedar.buffalo.edu/NIJ/data/signatures.rar
bsdtar -xf signatures.rar && rm signatures.rar && cd ..

# 4.（選用）SigNet 預訓練權重，只有深度特徵對照需要
.venv/bin/gdown 1l8NFdxSvQSLb2QTv71E6bKcTgvShKPpx -O sigver/models/signet.pth

# 5. 建特徵快取（教學版只需要不含 SIFT 的部分，約 6 分鐘）
.venv/bin/python features.py --no-sift

# 6. 啟動教學版，開 http://127.0.0.1:8000
.venv/bin/python app.py
```

鑑識版需要先建特徵快取與訓練模型：

```bash
.venv/bin/python features.py      # 2640 張抽 8 族特徵（約 30 分鐘，SIFT 佔 80%）
.venv/bin/python train.py         # 訓練 Elastic Net + 校準 → models/forensic.json
.venv/bin/python validate.py      # Cllr、Tippett 圖、逐族消融 → validation_report.png
.venv/bin/python forensic_app.py  # 開 http://127.0.0.1:8010
```

其他腳本：

```bash
.venv/bin/python handcrafted_demo.py   # 教學版的評估表與靜態報告圖
.venv/bin/python test_app.py           # 教學版端點測試（需先啟動 app.py）
```

## 檔案結構

```
features.py            共用：前處理 + 8 族特徵、CEDAR 快取（教學版與鑑識版都 import）
train.py               鑑識版：配對取樣、Elastic Net、LR 校準 → models/forensic.json
validate.py            鑑識版：Cllr、Tippett 圖、逐族消融
forensic_app.py        鑑識版後端（port 8010）
forensic.html          鑑識版前端
app.py                 教學版後端（port 8000，Python 內建 http.server，無框架）
index.html             教學版前端
handcrafted_demo.py    教學版：CEDAR 評估與靜態報告圖
signet_demo.py         SigNet 深度特徵對照
test_app.py            教學版端點測試
docs/SPEC.md           鑑識版規格
```

## 資料與授權

本 repo **不包含**任何資料集或模型權重，請依上述步驟自行取得：

- **CEDAR Signature Database**：研究用途，來源為 University at Buffalo，不可轉散布。
- **SigNet 預訓練權重**：來自 [luizgh/sigver](https://github.com/luizgh/sigver)，以 GPDS 資料集訓練，**限非商業用途**。
- **sigver**：BSD 3-clause。

## 已知限制

- **資料集與案件不符**：CEDAR 是西方拉丁字母簽名。中文簽名的筆畫數、結構、提筆次數分布都不同，σ 與權重必須用中文資料重新估計。
- **只看得到圖片**：離線方法看不出筆順、書寫速度、筆壓變化，而這些正是分辨「刻意模仿」最有用的線索。
- **校準資料規模小**：CEDAR 只有 55 位寫者，扣掉評估用的 30 人後，可用於訓練與校準的寫者相當有限，這會讓證據強度的絕對數值不穩定。
- **σ 借自他人**：參考樣本少於 10 張時，用的是族群的變動幅度，不是當事人自己的。
- **前處理假設乾淨影像**：白底、深色筆跡、已裁切。手機照片若光線不均或背景雜亂，二值化可能失敗。
