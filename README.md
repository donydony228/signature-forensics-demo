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

| 版本 | 狀態 | 說明 |
|---|---|---|
| **教學版**（`app.py`） | 已完成 | 用來說明整個流程：特徵 → 偏離幾個 σ → 距離門檻 → 二元判定 |
| **鑑識版**（`forensic_app.py`） | 規劃完成，未實作 | 改為差值向量 + 線性模型，輸出證據強度（likelihood ratio）而非二元判定 |

鑑識版的完整設計見 **[docs/SPEC.md](docs/SPEC.md)**。兩個版本的主要差異：

- 不給「本人／非本人」的結論，改報 likelihood ratio 加文字級距，由鑑識人員自行判斷
- 權重由資料學出來，而非所有特徵等權重
- 呈現的是「權重 × 差異」的證據分解，而不只是「差多少」
- 可匯出單一 HTML 報告，附進案件卷宗

## 方法概述

```
兩張簽名圖
  → 前處理：裁到筆跡範圍、等比縮放、Otsu 二值化
  → 特徵：幾何、方向分布、HOG、LBP 等
  → 每一維換算成「偏離幾個 σ」
  → 教學版：距離門檻 ｜ 鑑識版：線性模型 → likelihood ratio
  → 拆回具名特徵與熱區圖
```

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

# 5. 啟動教學版，開 http://127.0.0.1:8000
.venv/bin/python app.py
```

第一次執行會對 2640 張圖抽特徵並快取到 `data/`，約需一分鐘。

其他腳本：

```bash
.venv/bin/python handcrafted_demo.py   # 印出評估表，產生靜態報告圖
.venv/bin/python test_app.py           # 端點測試（需先啟動 app.py）
```

## 檔案結構

```
app.py                 教學版後端（Python 內建 http.server，無框架）
index.html             教學版前端（單一檔案，無外部函式庫）
handcrafted_demo.py    特徵計算、CEDAR 校準與評估、報告圖
signet_demo.py         SigNet 深度特徵對照，以及共用的資料集工具
test_app.py            端點測試
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
