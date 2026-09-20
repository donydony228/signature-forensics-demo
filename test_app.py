"""demo 網站端點測試：先在另一個終端機跑 .venv/bin/python app.py，再跑 .venv/bin/python test_app.py"""
import base64, json, urllib.request, urllib.error, io
from PIL import Image
BASE = 'http://127.0.0.1:8000'
S = str(__import__('pathlib').Path(__file__).parent / 'data/signatures')

def item(path, name=None):
    data = open(path, 'rb').read()
    return {'name': name or path.split('/')[-1], 'data': 'data:image/png;base64,' + base64.b64encode(data).decode()}

def post(body):
    req = urllib.request.Request(BASE + '/verify', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())

def get(path):
    try:
        with urllib.request.urlopen(BASE + path) as r: return r.status, r.headers['Content-Type'], len(r.read())
    except urllib.error.HTTPError as e: return e.code, None, 0

refs = [item(f'{S}/full_org/original_1_{i}.png') for i in (1, 2, 3)]
for label, q in (('genuine', f'{S}/full_org/original_1_20.png'), ('forged', f'{S}/full_forg/forgeries_1_5.png')):
    st, r = post({'references': refs, 'query': item(q)})
    print(label, st, 'accepted' if r['accepted'] else 'rejected', round(r['distance'], 2), '/', round(r['threshold'], 2),
          {k: round(v, 2) for k, v in r['groups'].items()}, 'fig', len(r['figure']))
    assert st == 200 and r['accepted'] == (label == 'genuine')

assert get('/')[0] == 200
assert get('/sample/original_1_1.png')[:2] == (200, 'image/png')
assert get('/sample/..%2F..%2Fapp.py')[0] == 404 and get('/sample/../app.py')[0] == 404, '範例路由不能讀到其他檔案'
assert post({'references': [], 'query': refs[0]})[0] == 400
assert post({'foo': 1})[0] == 400
assert post({'references': [{'name': 'x.png', 'data': base64.b64encode(b'not an image').decode()}], 'query': refs[0]})[0] == 422
blank = io.BytesIO(); Image.new('L', (300, 120), 255).save(blank, 'PNG')
assert post({'references': refs, 'query': {'name': 'blank.png', 'data': base64.b64encode(blank.getvalue()).decode()}})[0] == 422
# 手機照片情境：RGBA 透明背景 + 超大解析度
img = Image.open(f'{S}/full_org/original_1_20.png').convert('L')
big = img.resize((img.width * 4, img.height * 4))
rgba = Image.new('RGBA', big.size, (0, 0, 0, 0)); rgba.putalpha(Image.eval(big, lambda v: 255 - v))
buf = io.BytesIO(); rgba.save(buf, 'PNG')
st, r = post({'references': refs, 'query': {'name': 'big_rgba.png', 'data': base64.b64encode(buf.getvalue()).decode()}})
print('4x RGBA transparent genuine', st, r.get('accepted'), round(r.get('distance', -1), 2), r.get('error'))
assert st == 200 and r['accepted'], '放大、透明背景的真簽名應該判定為本人'
print('全部通過')
