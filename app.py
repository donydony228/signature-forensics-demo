"""簽名驗證 demo 網站：index.html 前端 + Python 內建 http.server 後端

判定：手工特徵（幾何 + HOG + LBP）和本人參考簽名的距離 ≤ 閾值 = 本人。σ 和閾值啟動時用 CEDAR 校準。

    .venv/bin/python app.py      # 開 http://127.0.0.1:8000
"""
import base64
import io
import json
import re
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import matplotlib.pyplot as plt

from handcrafted_demo import GEO_NAMES, GROUPS, N_REF, calibrate, plot_report, verify

ROOT = Path(__file__).parent
METHOD = '手工合併'  # calibrate() 結果裡的方法名稱：幾何 + HOG + LBP 合併
MAX_BODY = 50 * 1024 * 1024  # 上傳總大小上限
MAX_REFS = 20
SAMPLE_NAME = re.compile(r'^(original|forgeries)_\d{1,2}_\d{1,2}\.png$')  # 範例圖只開放 CEDAR 檔名，防路徑穿越

print('用 CEDAR 校準 σ 和閾值……')
SIGMAS, RESULTS = calibrate()
THRESHOLD = RESULTS[METHOD][3]


def run_verification(refs, query):
    """refs：[{name, data}]，query：{name, data}，data 是 base64 data URL。回傳給前端的 JSON"""
    with tempfile.TemporaryDirectory() as tmp:
        def save(folder, item):
            path = Path(tmp) / folder / Path(item['name']).name  # 只取檔名，丟掉前端傳來的任何路徑
            path.parent.mkdir()
            path.write_bytes(base64.b64decode(item['data'].split(',', 1)[-1]))
            return path

        ref_paths = [save(f'ref{i}', r) for i, r in enumerate(refs)]
        ref_mean_img, protos, (case,) = verify(ref_paths, [('待測', save('query', query))], SIGMAS)

    fig = plot_report(ref_mean_img, protos, len(refs), [case], THRESHOLD, bars=False)  # 長條改由網頁表格畫
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)

    distance = case['distance']
    return {
        'accepted': bool(distance <= THRESHOLD),
        'distance': distance,
        'threshold': THRESHOLD,
        'groups': dict(zip(GROUPS, case['groups'])),
        'features': [{'name': n, 'reference': protos[0][k], 'query': case['raw'][k], 'z': case['z'][0][k]}
                     for k, n in enumerate(GEO_NAMES)],
        'figure': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode(),
        'nRef': len(refs),
        'calibration': {'nRef': N_REF, 'eerGlobal': RESULTS[METHOD][0], 'eerUser': RESULTS[METHOD][1]},
    }


# ponytail: 單執行緒 HTTPServer，一次處理一個請求（matplotlib 本來就不是執行緒安全）；多人同時用再換 ThreadingHTTPServer + 鎖
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
            return self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        name = self.path.removeprefix('/sample/')
        if name != self.path and SAMPLE_NAME.match(name):
            path = ROOT / 'data/signatures' / ('full_org' if name.startswith('original') else 'full_forg') / name
            if path.exists():
                return self.send(200, path.read_bytes(), 'image/png')
        self.send_json(404, {'error': '找不到'})

    def do_POST(self):
        if self.path != '/verify':
            return self.send_json(404, {'error': '找不到'})
        length = int(self.headers.get('Content-Length', 0))
        if length > MAX_BODY:
            return self.send_json(413, {'error': f'上傳檔案總共超過 {MAX_BODY // 1024 // 1024} MB'})
        try:
            body = json.loads(self.rfile.read(length))
            refs, query = body['references'], body['query']
            if not 1 <= len(refs) <= MAX_REFS or not query:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return self.send_json(400, {'error': f'需要 1～{MAX_REFS} 張參考簽名和 1 張待測簽名'})
        try:
            self.send_json(200, run_verification(refs, query))
        except Exception as e:  # 壞掉的圖檔、整張空白找不到筆跡……都回給前端顯示，不讓伺服器掛掉
            self.log_error('verify failed: %r', e)
            self.send_json(422, {'error': f'無法處理這些圖片（{type(e).__name__}）。請確認是白底深色筆跡的簽名圖。'})


if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 8000), Handler)
    print('開啟 http://127.0.0.1:8000')
    server.serve_forever()
