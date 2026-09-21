"""鑑識版後端：forensic.html + Python 內建 http.server

輸出證據強度（likelihood ratio）與逐項證據分解，不給真偽結論。模型由 train.py 產生。

    .venv/bin/python forensic_app.py      # 開 http://127.0.0.1:8010
"""
import base64
import io
import json
import re
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

import features as F
import train as T

ROOT = Path(__file__).parent
PORT = 8010                      # 避開本機既有服務佔用的 8001
MAX_BODY = 80 * 1024 * 1024
MAX_SPECIMENS = 30
MIN_SPECIMENS = 3                # 低於這個數不計算，見 docs/SPEC.md 第 4 節
PERSONAL_SIGMA_FROM = 10         # 達到這個數才開始混入當事人自己的 σ
SIGMA_BLEND_K = 10               # 混合權重 n/(n+K)，n=10 時各半
SAMPLE_NAME = re.compile(r'^(original|forgeries)_\d{1,2}_\d{1,2}\.png$')   # 範例圖只開放 CEDAR 檔名

# 文字級距（ENFSI 風格）。使用單位若有既定結論用語，改這裡
BANDS = ((1000, '非常強烈支持'), (100, '強烈支持'), (10, '中度支持'), (2, '稍微支持'))
NEUTRAL_TEXT = '本次檢驗無法區分'

MODEL_PATH = ROOT / 'models/forensic.json'
if not MODEL_PATH.exists():
    raise SystemExit(f'缺模型檔 {MODEL_PATH.relative_to(ROOT)}。請依序執行：\n'
                     '  .venv/bin/python features.py   # 建特徵快取\n'
                     '  .venv/bin/python train.py      # 訓練模型\n'
                     '  .venv/bin/python validate.py   # 驗證（可選，但頁面會顯示驗證摘要）')
MODEL = json.loads(MODEL_PATH.read_text())
COEF = np.array(MODEL['coef'])
BASELINE = np.array(MODEL['baseline_diff'])
SIGMAS = {fam: np.array(v) for fam, v in MODEL['sigmas'].items()}
SLICES = {k: tuple(v) for k, v in MODEL['slices'].items()}
NAMED = [(fam, i, name) for fam in F.ALL_FAMILIES if F.FEATURE_NAMES[fam]
         for i, name in enumerate(F.FEATURE_NAMES[fam])]


def band(lr):
    """LR → 文字級距"""
    x = lr if lr >= 1 else 1 / lr
    if x < BANDS[-1][0]:
        return {'text': NEUTRAL_TEXT, 'direction': '', 'neutral': True}
    text = next(t for threshold, t in BANDS if x >= threshold)
    return {'text': text, 'direction': '同一人所寫' if lr >= 1 else '不同人所寫', 'neutral': False}


def to_lr(score):
    cal = MODEL['calibration']
    return float(np.exp(np.clip(cal['a'] * score + cal['b'], -30, 30)))


def blend_sigmas(ref_feats):
    """σ 策略：樣本少時用族群變化範圍，多時混入當事人自己的。回傳 (sigmas, 說明文字)"""
    n = len(ref_feats)
    if n < PERSONAL_SIGMA_FROM:
        return SIGMAS, (f'σ 來源：族群變化範圍（開發資料集 {len(MODEL["train_users"]) + len(MODEL["calib_users"])} '
                        f'位書寫者）。已知樣本 {n} 張，未達 {PERSONAL_SIGMA_FROM} 張，'
                        '個人變化範圍樣本不足，尚未採用當事人自己的變化範圍。')
    w = n / (n + SIGMA_BLEND_K)
    out = {}
    for fam in F.FAMILIES:
        personal = np.stack([f[fam] for f in ref_feats]).std(axis=0, ddof=1)
        out[fam] = np.sqrt(w * personal ** 2 + (1 - w) * SIGMAS[fam] ** 2)
        out[fam] = np.maximum(out[fam], SIGMAS[fam] * 0.2)   # 個人 σ 偶爾極小，設個下限避免 z 爆掉
    return out, (f'σ 來源：當事人自己的變化範圍佔 {w:.0%}、族群變化範圍佔 {1 - w:.0%}'
                 f'（已知樣本 {n} 張）。')


def cell_grid(contributions):
    """HOG 的貢獻攤回 15×22 格：每格把 9 個方向加總"""
    lo, hi = SLICES['HOG']
    return contributions[lo:hi].reshape(*F.HOG_GRID, F.HOG_BINS).sum(axis=2)


def png_data_url(aligned):
    """對齊後的反色圖（墨跡亮）→ 一般觀感的 PNG（墨跡深、背景白）"""
    img = Image.fromarray((255 - aligned).astype(np.uint8), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def run_comparison(specimens, questioned):
    """specimens / questioned：[{name, data}]。回傳給前端的 JSON"""
    with tempfile.TemporaryDirectory() as tmp:
        def save(folder, item):
            path = Path(tmp) / folder / Path(item['name']).name    # 只取檔名，丟掉前端傳來的任何路徑
            path.parent.mkdir()
            path.write_bytes(base64.b64decode(item['data'].split(',', 1)[-1]))
            return path

        refs = [F.extract(save(f'ref{i}', s)) for i, s in enumerate(specimens)]
        q_feats, q_aligned, q_sift = F.extract(save('questioned', questioned))

    ref_feats = [r[0] for r in refs]
    sigmas, sigma_note = blend_sigmas(ref_feats)

    # 每張已知樣本各算一次：差值向量 → 分數 → LR
    # sift_pair 的第一個參數與訓練時一致（訓練時固定是本人的真簽名）
    diffs = np.array([T.diff_vector(q_feats, rf, sigmas, F.sift_pair(rs, q_sift))
                      for (rf, _, rs) in refs])
    scores = diffs @ COEF + MODEL['intercept']
    per_specimen = [{'name': s['name'], 'score': float(sc), 'lr': to_lr(sc), 'band': band(to_lr(sc))}
                    for s, sc in zip(specimens, scores)]

    # 標題數字：分數取平均（模型是線性的，等於差值向量取平均後算一次）
    mean_diff = diffs.mean(axis=0)
    score = float(mean_diff @ COEF + MODEL['intercept'])
    lr = to_lr(score)

    # 證據分解：貢獻 = 權重 ×（實際差異 − 同源配對的典型差異）
    contributions = COEF * (mean_diff - BASELINE)
    families = {fam: float(contributions[slice(*SLICES[fam])].sum()) for fam in F.ALL_FAMILIES}

    # 具名特徵：取貢獻最大的 10 項，附帶「已知樣本平均值」與帶正負號的偏離
    ref_mean = {fam: np.mean([f[fam] for f in ref_feats], axis=0) for fam in F.FAMILIES}
    named = []
    for fam, i, name in NAMED:
        c = float(contributions[SLICES[fam][0] + i])
        if fam == 'SIFT':                                      # SIFT 是成對算的，沒有「本人平均」
            named.append({'name': name, 'family': fam, 'reference': None,
                          'questioned': float(diffs[:, SLICES[fam][0] + i].mean()), 'z': None, 'contribution': c})
        else:
            named.append({'name': name, 'family': fam, 'reference': float(ref_mean[fam][i]),
                          'questioned': float(q_feats[fam][i]),
                          'z': float((q_feats[fam][i] - ref_mean[fam][i]) / sigmas[fam][i]),
                          'contribution': c})
    named.sort(key=lambda f: -abs(f['contribution']))

    grid = cell_grid(contributions)
    mean_ref_aligned = np.mean([r[1] for r in refs], axis=0)
    return {
        'lr': lr, 'score': score, 'band': band(lr),
        'perSpecimen': per_specimen,
        'families': families,
        'named': named[:10],
        'heatmap': {'grid': grid.tolist(), 'max': float(np.abs(grid).max()),
                    'rows': F.HOG_GRID[0], 'cols': F.HOG_GRID[1], 'cell': F.HOG_CELL},
        'images': {'reference': png_data_url(mean_ref_aligned), 'questioned': png_data_url(q_aligned)},
        'sigmaNote': sigma_note,
        'nSpecimens': len(specimens),
        'provenance': MODEL.get('validation'),
        'dataset': MODEL['dataset'],
        'hyperparams': MODEL['hyperparams'],
        'dims': {'total': int(len(COEF)), 'nonzero': int((COEF != 0).sum())},
    }


# ponytail: 單執行緒 HTTPServer，一次處理一個請求；多人同時用再換 ThreadingHTTPServer + 鎖
class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, obj):
        self.send(status, json.dumps(obj, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

    def do_GET(self):
        if self.path == '/':
            return self.send(200, (ROOT / 'forensic.html').read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/model':
            return self.send_json(200, {'validation': MODEL.get('validation'), 'dataset': MODEL['dataset'],
                                        'families': list(F.ALL_FAMILIES),
                                        'dims': {'total': int(len(COEF)), 'nonzero': int((COEF != 0).sum())},
                                        'minSpecimens': MIN_SPECIMENS,
                                        'personalSigmaFrom': PERSONAL_SIGMA_FROM})
        name = self.path.removeprefix('/sample/')
        if name != self.path and SAMPLE_NAME.match(name):
            path = F.CEDAR / ('full_org' if name.startswith('original') else 'full_forg') / name
            if path.exists():
                return self.send(200, path.read_bytes(), 'image/png')
        self.send_json(404, {'error': '找不到'})

    def do_POST(self):
        if self.path != '/compare':
            return self.send_json(404, {'error': '找不到'})
        length = int(self.headers.get('Content-Length', 0))
        if length > MAX_BODY:
            return self.send_json(413, {'error': f'上傳檔案總共超過 {MAX_BODY // 1024 // 1024} MB'})
        try:
            body = json.loads(self.rfile.read(length))
            specimens, questioned = body['specimens'], body['questioned']
            if not isinstance(specimens, list) or not questioned:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return self.send_json(400, {'error': '請提供已知樣本與待鑑簽名'})
        if not MIN_SPECIMENS <= len(specimens) <= MAX_SPECIMENS:
            return self.send_json(400, {
                'error': f'需要 {MIN_SPECIMENS}～{MAX_SPECIMENS} 張已知樣本（目前 {len(specimens)} 張）。'
                         f'低於 {MIN_SPECIMENS} 張無法估計變化範圍，不予計算。'})
        try:
            self.send_json(200, run_comparison(specimens, questioned))
        except Exception as e:      # 壞掉的圖檔、整張空白找不到筆跡……回給前端顯示，不讓伺服器掛掉
            self.log_error('compare failed: %r', e)
            self.send_json(422, {'error': f'無法處理這些圖片（{type(e).__name__}）。'
                                          '請確認是白底深色筆跡、已裁切到簽名範圍的影像。'})


if __name__ == '__main__':
    v = MODEL.get('validation')
    print(f'模型：{len(COEF)} 維，非零權重 {int((COEF != 0).sum())}')
    if v:
        print(f'驗證：Cllr 熟練偽造 {v["cllr"]["熟練偽造"]:.3f}／隨機不同人 {v["cllr"]["隨機不同人"]:.3f}')
    else:
        print('警告：模型尚未驗證，請先執行 validate.py')
    print(f'開啟 http://127.0.0.1:{PORT}')
    HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
