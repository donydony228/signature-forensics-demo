"""共用特徵計算：前處理 + 8 族特徵。教學版與鑑識版都 import 這個模組。

鑑識用途最怕「報告裡的數字算法和驗證時不一致」，共用一份程式碼是唯一的保證。

特徵族（依 docs/SPEC.md 第 3 節）：

| 族 | 維度 | 算在哪張圖上 |
|---|---|---|
| 幾何 geometric | 7 | 尺寸正規化後的原圖 |
| 方向分布 direction | 10 | 同上（骨架） |
| 小波 wavelet | 10 | 對齊後的 150×220 |
| Radon radon | 36 | 同上 |
| GLCM glcm | 48 | 同上 |
| HOG hog | 2970 | 同上 |
| LBP lbp | 250 | 同上 |
| SIFT sift | 3 | 成對計算，見 sift_pair() |

    .venv/bin/python features.py          # 建立 CEDAR 特徵快取（SIFT 那部分很慢，約 20 分鐘）
    .venv/bin/python features.py --no-sift
"""
import sys
from pathlib import Path

import numpy as np
import pywt
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, transform
from skimage.feature import SIFT, graycomatrix, graycoprops, hog, local_binary_pattern, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform
from sklearn.metrics import roc_curve

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'sigver'))
from sigver.preprocessing.normalize import preprocess_signature  # noqa: E402  只用 skimage，不含 torch

# ---- 前處理 ------------------------------------------------------------------
CANVAS = (952, 1360)                                   # sigver 前處理的畫布，簽名要比它小
SIZE_BOX = (int(0.75 * CANVAS[0]), int(0.75 * CANVAS[1]))  # 每張簽名等比縮放到剛好塞進這個框
ALIGNED = (150, 220)                                   # 對齊後的圖，HOG/LBP/小波/Radon/GLCM 都用它

# ---- 各族的參數 --------------------------------------------------------------
HOG_CELL = 10
HOG_GRID = (ALIGNED[0] // HOG_CELL, ALIGNED[1] // HOG_CELL)   # 15×22 格
HOG_BINS = 9
LBP_GRID = (5, 5)
LBP_BINS = 10                                          # P=8, R=1 的 uniform 模式數
RADON_ANGLES = np.arange(0, 180, 5, dtype=float)       # 36 個角度
GLCM_DISTANCES = (1, 3)
GLCM_ANGLES = (0, 45, 90, 135)
GLCM_PROPS = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')
GLCM_LEVELS = 32
WAVELET, WAVELET_LEVEL = 'db2', 3
SIFT_MAX_KEYPOINTS = 600      # 全存會讓快取膨脹到 GB 級；固定種子抽樣，配對率的分母也跟著一致

# 骨架的 8 鄰居位移，順時針從左上開始（用來數端點與交叉點）
_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
DIRECTION_BINS = 8                                     # 0°~180° 切 8 份，每份 22.5°
DIRECTION_WINDOW = 7                                   # 局部主軸的視窗邊長（半徑 3）

FAMILIES = ('幾何', '方向分布', '小波', 'Radon', 'GLCM', 'HOG', 'LBP')  # SIFT 是成對的，另外算
SIFT_FAMILY = 'SIFT'
ALL_FAMILIES = FAMILIES + (SIFT_FAMILY,)

N_USERS, N_SIGS = 55, 24
CEDAR = ROOT / 'data/signatures'


def _glcm_names():
    return [f'GLCM {p}（距離 {d}, {a}°）' for p in GLCM_PROPS for d in GLCM_DISTANCES for a in GLCM_ANGLES]


def _wavelet_names():
    bands = ('水平', '垂直', '對角')
    return ['小波 LL 能量'] + [f'小波 第 {lv} 層 {b}' for lv in range(WAVELET_LEVEL, 0, -1) for b in bands]


# 具名特徵的名稱；HOG 與 LBP 的個別維度沒有意義，以熱區呈現
FEATURE_NAMES = {
    '幾何': ['寬高比', '墨跡密度', '質心水平位置', '質心垂直位置', '整體傾角（度）', '筆畫段數', '筆畫總長 ÷ 高度'],
    '方向分布': [f'筆畫方向 {int(i * 180 / DIRECTION_BINS)}° 比例' for i in range(DIRECTION_BINS)]
                + ['端點數 ÷ 骨架長', '交叉點數 ÷ 骨架長'],
    '小波': _wavelet_names(),
    'Radon': [f'Radon 投影 {int(a)}°' for a in RADON_ANGLES],
    'GLCM': _glcm_names(),
    'HOG': None,
    'LBP': None,
    'SIFT': ['SIFT 配對率', 'SIFT 配對點平均距離', 'SIFT 幾何一致性比例'],
}
DIMS = {
    '幾何': 7, '方向分布': DIRECTION_BINS + 2, '小波': 3 * WAVELET_LEVEL + 1,
    'Radon': len(RADON_ANGLES), 'GLCM': len(GLCM_PROPS) * len(GLCM_DISTANCES) * len(GLCM_ANGLES),
    'HOG': HOG_GRID[0] * HOG_GRID[1] * HOG_BINS, 'LBP': LBP_GRID[0] * LBP_GRID[1] * LBP_BINS, 'SIFT': 3,
}


# ---- 影像載入與正規化 --------------------------------------------------------
def load_gray(path):
    """任意圖檔 → 灰階 uint8，並統一簽名大小。處理手機照片的 EXIF 旋轉、PNG 透明背景。

    尺寸正規化：裁到筆跡外框，再等比縮放到 SIZE_BOX。不同照片、掃描器的解析度不同，
    不統一的話，同一個簽名大小不同，HOG/LBP 就對不上同一組格子。"""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        img = Image.alpha_composite(Image.new('RGBA', img.size, 'white'), img)
    img = img.convert('L')

    gray = np.asarray(img)
    rows, cols = np.nonzero(gray < filters.threshold_otsu(gray))
    if rows.size == 0:
        raise ValueError('找不到筆跡')
    # 外框取 0.5%～99.5% 分位數：解析度未知，沒辦法用固定像素數去雜點，改成忽略最外圍零星的點
    (r0, r1), (c0, c1) = np.percentile(rows, [0.5, 99.5]), np.percentile(cols, [0.5, 99.5])
    pad = 0.03 * max(r1 - r0, c1 - c0)
    img = img.crop((max(c0 - pad, 0), max(r0 - pad, 0), min(c1 + pad, img.width), min(r1 + pad, img.height)))
    scale = min(SIZE_BOX[0] / img.height, SIZE_BOX[1] / img.width)
    return np.asarray(img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS))


def ink_mask(gray):
    """Otsu 二值化取出墨跡，刪掉 20 px 以下的雜點"""
    return morphology.remove_small_objects(gray < filters.threshold_otsu(gray), max_size=20)


def aligned_image(gray):
    """置中到畫布再縮放裁切成 150×220（反色圖：墨跡亮、背景 0），HOG 等對齊型特徵用它"""
    return preprocess_signature(gray, CANVAS, input_size=ALIGNED)


# ---- 幾何 --------------------------------------------------------------------
def geometric(gray):
    """7 維，算在原圖上：前處理會把簽名置中縮放，寬高比、重心這類資訊要看原圖才準"""
    ink = ink_mask(gray)
    rows, cols = np.nonzero(ink)
    h, w = np.ptp(rows) + 1, np.ptp(cols) + 1
    orientation = measure.regionprops(ink.astype(int))[0].orientation  # 整個簽名主軸方向，弧度
    return np.array([
        w / h,
        ink.sum() / (w * h),
        (cols.mean() - cols.min()) / w,
        (rows.mean() - rows.min()) / h,
        90 - abs(np.degrees(orientation)),          # 主軸跟水平線的夾角，0 = 完全水平
        measure.label(ink).max(),                   # 連通塊數 ≈ 提筆次數 + 1
        morphology.skeletonize(ink).sum() / h,      # 骨架長度（不受筆粗影響）÷ 高度（不受解析度影響）
    ], dtype=float)


# ---- 方向分布 ----------------------------------------------------------------
def _local_orientation(skel):
    """每個骨架點的筆畫走向（度，0~180）。取視窗內骨架點座標的共變異數主軸。

    不用「8 鄰居兩點連線」的算法：那種算法可得的角度只有少數幾種，
    22.5°~45° 以外的某些區間（67.5°~90°、157.5°~180°）結構上永遠取不到值。"""
    m = skel.astype(float)
    ys, xs = np.indices(skel.shape).astype(float)
    win = DIRECTION_WINDOW
    s = ndi.uniform_filter(m, win)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my = ndi.uniform_filter(m * xs, win) / s, ndi.uniform_filter(m * ys, win) / s
        cxx = ndi.uniform_filter(m * xs * xs, win) / s - mx * mx
        cyy = ndi.uniform_filter(m * ys * ys, win) / s - my * my
        cxy = ndi.uniform_filter(m * xs * ys, win) / s - mx * my
        return np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy)) % 180


def direction(gray):
    """10 維：骨架的筆畫走向分布（8 箱）+ 端點數、交叉點數。

    和 HOG 不同：HOG 用影像梯度，方向垂直於筆畫邊緣，且受筆粗影響；
    這裡量的是骨架的切線方向，也就是「筆畫往哪裡走」。"""
    skel = morphology.skeletonize(ink_mask(gray))
    if skel.sum() == 0:
        return np.zeros(DIMS['方向分布'])

    angles = _local_orientation(skel)[skel]
    angles = angles[np.isfinite(angles)]
    bins = (angles / (180 / DIRECTION_BINS)).astype(int) % DIRECTION_BINS
    hist = np.bincount(bins, minlength=DIRECTION_BINS).astype(float)

    # 端點與交叉點：數 8 鄰居裡有幾個骨架點
    n_neighbors = np.zeros(skel.shape, dtype=np.uint8)
    for dy, dx in _OFFSETS:
        n_neighbors += (np.roll(np.roll(skel, dy, axis=0), dx, axis=1) & skel).astype(np.uint8)

    return np.concatenate([
        hist / hist.sum() if hist.sum() else hist,
        [(n_neighbors[skel] == 1).sum() / skel.sum(),    # 端點：一筆的起點或終點
         (n_neighbors[skel] >= 3).sum() / skel.sum()],   # 交叉點：筆畫相交或分叉
    ])


# ---- 小波 / Radon / GLCM -----------------------------------------------------
def wavelet(aligned):
    """10 維：3 層小波分解後各子帶的能量佔比，描述細節集中在哪個尺度"""
    coeffs = pywt.wavedec2(aligned.astype(float) / 255, WAVELET, level=WAVELET_LEVEL)
    energies = [np.mean(coeffs[0] ** 2)]
    for detail in coeffs[1:]:
        energies.extend(np.mean(band ** 2) for band in detail)
    energies = np.array(energies)
    return energies / energies.sum() if energies.sum() else energies


def radon(aligned):
    """36 維：各角度投影的能量佔比。筆畫整體走向會讓某些角度的投影特別集中"""
    sino = transform.radon(aligned.astype(float) / 255, theta=RADON_ANGLES, preserve_range=True)
    energy = (sino ** 2).sum(axis=0)
    return energy / energy.sum() if energy.sum() else energy


def glcm(aligned):
    """48 維：灰階共生矩陣的 6 個統計量 × 2 個距離 × 4 個角度，描述筆觸的紋理"""
    q = (aligned.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    m = graycomatrix(q, distances=GLCM_DISTANCES, angles=np.radians(GLCM_ANGLES),
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    # correlation 在方差為 0 的區塊會是 NaN
    return np.nan_to_num(np.concatenate([graycoprops(m, p).ravel() for p in GLCM_PROPS]))


# ---- HOG / LBP ---------------------------------------------------------------
def hog_lbp(aligned):
    """HOG 2970 維（15×22 格 × 9 方向）與 LBP 250 維（5×5 格 × 10 模式）"""
    h = hog(aligned, orientations=HOG_BINS, pixels_per_cell=(HOG_CELL, HOG_CELL),
            cells_per_block=(1, 1), feature_vector=True)
    codes = local_binary_pattern(aligned, P=8, R=1, method='uniform')
    lbp = [np.bincount(cell.ravel().astype(int), minlength=LBP_BINS) / cell.size
           for band in np.array_split(codes, LBP_GRID[0], axis=0)
           for cell in np.array_split(band, LBP_GRID[1], axis=1)]
    return h, np.concatenate(lbp)


# ---- SIFT（成對）-------------------------------------------------------------
def sift_extract(gray):
    """回傳 (keypoints float32 (N,2), descriptors uint8 (N,128))。
    單張圖算不出特徵值，要兩張一起比才有意義，見 sift_pair()"""
    detector = SIFT()
    try:
        detector.detect_and_extract(gray.astype(float) / 255)
    except RuntimeError:                                  # 筆跡太少，找不到關鍵點
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    kp, desc = detector.keypoints.astype(np.float32), detector.descriptors
    if len(kp) > SIFT_MAX_KEYPOINTS:
        pick = np.random.default_rng(0).choice(len(kp), SIFT_MAX_KEYPOINTS, replace=False)
        pick.sort()
        kp, desc = kp[pick], desc[pick]
    return kp, desc


def sift_pair(a, b):
    """3 維：配對率、配對點的平均描述子距離、通過仿射 RANSAC 的比例。

    第三維特別重要：熟練偽造常見的模式是「局部像、整體比例錯」，
    會表現為「配對數多但通不過幾何檢驗」，單看配對數抓不到。"""
    (kp_a, d_a), (kp_b, d_b) = a, b
    if len(d_a) < 4 or len(d_b) < 4:
        return np.array([0.0, 2.0, 0.0])                  # 平均距離給最大值 2（單位化後的上限）

    unit_a = d_a / (np.linalg.norm(d_a, axis=1, keepdims=True) + 1e-9)
    unit_b = d_b / (np.linalg.norm(d_b, axis=1, keepdims=True) + 1e-9)
    matches = match_descriptors(unit_a, unit_b, metric='euclidean', cross_check=True, max_ratio=0.8)
    if len(matches) < 3:
        return np.array([len(matches) / min(len(d_a), len(d_b)), 2.0, 0.0])

    rate = len(matches) / min(len(d_a), len(d_b))
    mean_dist = np.linalg.norm(unit_a[matches[:, 0]] - unit_b[matches[:, 1]], axis=1).mean()
    src = kp_a[matches[:, 0]][:, ::-1]                    # keypoints 是 (row, col)，轉成 (x, y)
    dst = kp_b[matches[:, 1]][:, ::-1]
    try:
        _, inliers = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=8, max_trials=200)
        consistency = float(inliers.mean()) if inliers is not None else 0.0
    except Exception:                                     # 配對點共線等退化情況
        consistency = 0.0
    return np.array([rate, mean_dist, consistency])


# ---- 一張圖的全部特徵 --------------------------------------------------------
def extract(path, with_sift=True):
    """回傳 (特徵字典, 對齊後的圖, SIFT 的 (keypoints, descriptors) 或 None)"""
    gray = load_gray(path)
    aligned = aligned_image(gray)
    h, lbp = hog_lbp(aligned)
    feats = {
        '幾何': geometric(gray), '方向分布': direction(gray),
        '小波': wavelet(aligned), 'Radon': radon(aligned), 'GLCM': glcm(aligned),
        'HOG': h, 'LBP': lbp,
    }
    return feats, aligned, sift_extract(gray) if with_sift else None


# ---- CEDAR 資料集 ------------------------------------------------------------
def cedar_path(kind, user, i):   # kind: 'original' | 'forgeries'，user/i 從 1 開始
    folder = 'full_org' if kind == 'original' else 'full_forg'
    return CEDAR / folder / f'{kind}_{user}_{i}.png'


def cedar_index():
    """(kind, user, i) 的固定順序，快取的列順序就是這個"""
    return [(kind, u, i) for kind in ('original', 'forgeries')
            for u in range(1, N_USERS + 1) for i in range(1, N_SIGS + 1)]


def cedar_features():
    """回傳 {族: {kind: (55, 24, 維度)}}，缺快取時報錯（請先跑 features.py）"""
    cache = ROOT / 'data/cedar_features.npz'
    if not cache.exists():
        raise FileNotFoundError(f'缺特徵快取 {cache}，請先執行：.venv/bin/python features.py')
    d = np.load(cache)
    return {fam: {kind: d[f'{kind}_{fam}'].reshape(N_USERS, N_SIGS, -1) for kind in ('original', 'forgeries')}
            for fam in FAMILIES}


def cedar_sift():
    """回傳 {(kind, user, i): (keypoints, descriptors)}，user/i 為 1 起算的整數"""
    cache = ROOT / 'data/cedar_sift.npz'
    if not cache.exists():
        raise FileNotFoundError(f'缺 SIFT 快取 {cache}，請先執行：.venv/bin/python features.py')
    d = np.load(cache)
    out = {}
    for key in d.files:
        kind, u, i, what = key.split('|')
        if what == 'kp':
            out[(kind, int(u), int(i))] = (d[key], d[f'{kind}|{u}|{i}|desc'])
    return out


def build_cache(with_sift=True):
    """對 2640 張圖抽特徵並快取。SIFT 很慢（約 0.5 秒／張），所以分成兩個檔案"""
    index = cedar_index()
    rows = {fam: [] for fam in FAMILIES}
    sift_store = {}
    for n, (kind, u, i) in enumerate(index, 1):
        feats, _, sift = extract(cedar_path(kind, u, i), with_sift=with_sift)
        for fam, v in feats.items():
            rows[fam].append(v)
        if with_sift:                                  # 分開存：兩者第一維相同，湊成 object array 會被當成廣播
            sift_store[f'{kind}|{u}|{i}|kp'] = sift[0]
            sift_store[f'{kind}|{u}|{i}|desc'] = sift[1]
        if n % 200 == 0 or n == len(index):
            print(f'  {n}/{len(index)}', flush=True)

    half = len(index) // 2
    out = {}
    for fam, vals in rows.items():
        stacked = np.stack(vals)
        out[f'original_{fam}'], out[f'forgeries_{fam}'] = stacked[:half], stacked[half:]
    np.savez(ROOT / 'data/cedar_features.npz', **out)
    print(f'已存 data/cedar_features.npz（各族維度：{ {f: out[f"original_{f}"].shape[1] for f in FAMILIES} }）')
    if with_sift:
        np.savez_compressed(ROOT / 'data/cedar_sift.npz', **sift_store)
        size = (ROOT / 'data/cedar_sift.npz').stat().st_size / 1e6
        print(f'已存 data/cedar_sift.npz（{size:.0f} MB）')


# ---- 共用統計 ----------------------------------------------------------------
def pooled_sigma(genuine, floor):
    """人內變異：每個人自己簽名之間的標準差，再對所有人取平均（均方根）。
    genuine 形狀 (人數, 每人張數, 維度)。

    floor：HOG/LBP 等大量空白維度的 σ 會是 0，除下去會爆掉，下限設為非零 σ 的中位數。
    幾何等具名特徵不設下限：各維單位不同，共用下限會把小單位的維度（墨跡密度）壓成 0σ"""
    sigma = np.sqrt((genuine.std(axis=1) ** 2).mean(axis=0))
    return np.maximum(sigma, np.median(sigma[sigma > 0])) if floor else sigma


def eer(genuine_scores, forgery_scores):
    """等錯誤率：誤拒率 = 誤收率 的那個點。回傳 (EER, 閾值)"""
    y = np.r_[np.ones(len(genuine_scores)), np.zeros(len(forgery_scores))]
    fpr, tpr, thr = roc_curve(y, np.r_[genuine_scores, forgery_scores])
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    return (fpr[i] + 1 - tpr[i]) / 2, thr[i]


if __name__ == '__main__':
    print(f'抽 CEDAR 特徵（{N_USERS} 人 × {N_SIGS} 張 × 真偽兩類 = {2 * N_USERS * N_SIGS} 張）')
    build_cache(with_sift='--no-sift' not in sys.argv)
