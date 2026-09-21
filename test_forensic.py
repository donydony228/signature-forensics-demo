"""鑑識版端點測試

先在另一個終端機跑 .venv/bin/python forensic_app.py，再跑 .venv/bin/python test_forensic.py
"""
import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

BASE = 'http://127.0.0.1:8010'
SIGS = Path(__file__).parent / 'data/signatures'


def item(path, name=None):
    return {'name': name or Path(path).name,
            'data': 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()}


def post(body, route='/compare'):
    req = urllib.request.Request(BASE + route, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(route):
    try:
        with urllib.request.urlopen(BASE + route) as r:
            return r.status, r.headers['Content-Type'], len(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None, 0


specimens = [item(SIGS / 'full_org' / f'original_1_{i}.png') for i in range(1, 6)]

# 真簽名與熟練偽造：LR 應該分別偏向兩個方向
results = {}
for label, path in (('genuine', SIGS / 'full_org/original_1_20.png'),
                    ('forged', SIGS / 'full_forg/forgeries_1_5.png')):
    st, r = post({'specimens': specimens, 'questioned': item(path)})
    assert st == 200, (st, r)
    results[label] = r
    fams = ' '.join(f'{k} {v:+.2f}' for k, v in r['families'].items())
    print(f'{label:<8} LR {r["lr"]:>10.3g}  {r["band"]["text"]}  分數 {r["score"]:+.3f}')
    print(f'         族貢獻 {fams}')
    print(f'         逐張 LR ' + ' '.join(f'{p["lr"]:.3g}' for p in r['perSpecimen']))
    top = r['named'][0]
    print(f'         最大貢獻的具名特徵：{top["name"]} {top["contribution"]:+.3f}')

assert results['genuine']['lr'] > results['forged']['lr'], '真簽名的 LR 應該高於偽造'

# 證據分解要和分數一致：Σ 貢獻 + 基準項 = 分數（檢查兩者差值是常數）
g, f = results['genuine'], results['forged']
resid = [r['score'] - sum(r['families'].values()) for r in (g, f)]
assert abs(resid[0] - resid[1]) < 1e-6, f'貢獻拆解與分數不一致：{resid}'
print(f'\n貢獻拆解檢查：分數 − Σ族貢獻 = {resid[0]:.4f}（兩案一致，代表是同一個基準常數）')

# 熱區維度
hm = g['heatmap']
assert len(hm['grid']) == hm['rows'] and len(hm['grid'][0]) == hm['cols']
print(f'熱區 {hm["rows"]}×{hm["cols"]} 格，色階上限 ±{hm["max"]:.3f}')

# 錯誤處理
print()
print('樣本不足 2 張 ', post({'specimens': specimens[:2], 'questioned': specimens[0]})[0])
assert post({'specimens': specimens[:2], 'questioned': specimens[0]})[0] == 400
print('缺欄位      ', post({'foo': 1})[0])
assert post({'foo': 1})[0] == 400
bad = {'name': 'x.png', 'data': base64.b64encode(b'not an image').decode()}
print('壞圖檔      ', post({'specimens': specimens, 'questioned': bad})[0])
assert post({'specimens': specimens, 'questioned': bad})[0] == 422
blank = io.BytesIO()
Image.new('L', (300, 120), 255).save(blank, 'PNG')
blank_item = {'name': 'blank.png', 'data': base64.b64encode(blank.getvalue()).decode()}
print('空白圖      ', post({'specimens': specimens, 'questioned': blank_item})[0])
assert post({'specimens': specimens, 'questioned': blank_item})[0] == 422

print('\n首頁        ', get('/'))
print('模型資訊    ', get('/model'))
print('範例圖      ', get('/sample/original_1_1.png'))
print('路徑穿越    ', get('/sample/../forensic_app.py')[0], get('/sample/..%2Fforensic_app.py')[0])
assert get('/')[0] == 200 and get('/model')[0] == 200
assert get('/sample/../forensic_app.py')[0] == 404, '範例路由不能讀到其他檔案'

# 10 張以上要改用混合 σ
many = [item(SIGS / 'full_org' / f'original_1_{i}.png') for i in range(1, 13)]
st, r = post({'specimens': many, 'questioned': item(SIGS / 'full_org/original_1_20.png')})
assert st == 200 and '當事人自己' in r['sigmaNote'], r['sigmaNote']
print(f'\n12 張已知樣本：{r["sigmaNote"]}')
print(f'  LR {r["lr"]:.3g}（5 張時是 {results["genuine"]["lr"]:.3g}）')

print('\n全部通過')
