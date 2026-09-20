"""手工特徵簽名驗證示範（CEDAR）—— 不用任何神經網路

流程：簽名圖 → 手工特徵（幾何、HOG、LBP）→ 特徵向量 → 和本人參考簽名比距離 → 距離 ≤ 閾值 = 本人

每個特徵的差距用「偏離幾個 σ」表示。σ 是「一般人自己簽名時，這個特徵正常會變動多少」，
從另外 25 人（不參與評估）的真簽名估計出來。本人只有 3 張參考簽名，自己的 σ 估不準。
距離可以拆回每個特徵，HOG 的每一維對應圖上的一個 10×10 格子，所以能精確畫回圖上。

    .venv/bin/python handcrafted_demo.py      # 印出評估表和特徵差距，存 handcrafted_report.png
"""
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageOps
from skimage import filters, measure, morphology
from skimage.feature import hog, local_binary_pattern

from signet_demo import CANVAS, N_SIGS, N_USERS, cedar_features, cedar_path, eer
from sigver.preprocessing.normalize import preprocess_signature

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).parent
GEO_NAMES = ['寬高比', '墨跡密度', '質心水平位置', '質心垂直位置', '整體傾角（度）', '筆畫段數', '筆畫總長 ÷ 高度']
HOG_CELL = 10                    # 150×220 的圖切成 15×22 格，每格 9 個梯度方向
HOG_GRID = (150 // HOG_CELL, 220 // HOG_CELL)
LBP_GRID = (5, 5)                # LBP 紋理直方圖，切 5×5 大格
GROUPS = ('幾何', 'HOG', 'LBP')
EVAL_USERS, N_REF = range(30), 3


SIZE_BOX = (int(0.75 * CANVAS[0]), int(0.75 * CANVAS[1]))  # 每張簽名等比縮放到剛好塞進這個框


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


def geometric(gray):
    """在原圖上算：前處理會把簽名置中縮放，寬高比、大小這類資訊要看原圖才準"""
    ink = gray < filters.threshold_otsu(gray)
    ink = morphology.remove_small_objects(ink, max_size=20)  # 去掉掃描雜點（20 px 以下）
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


def texture(processed):
    """在前處理後的 150×220 圖上算：大家對齊到同一個格子，HOG/LBP 的每一維才對得起來"""
    h = hog(processed, orientations=9, pixels_per_cell=(HOG_CELL, HOG_CELL),
            cells_per_block=(1, 1), feature_vector=True)
    codes = local_binary_pattern(processed, P=8, R=1, method='uniform')  # 10 種模式
    lbp = [np.bincount(cell.ravel().astype(int), minlength=10) / cell.size
           for band in np.array_split(codes, LBP_GRID[0], axis=0)
           for cell in np.array_split(band, LBP_GRID[1], axis=1)]
    return h, np.concatenate(lbp)


def extract(path):
    gray = load_gray(path)
    processed = preprocess_signature(gray, CANVAS)
    return (geometric(gray), *texture(processed)), processed


def cedar_handcrafted():
    """2640 張全部抽特徵，快取。回傳 {group: (genuine, forgery)}，形狀 (55, 24, 維度)"""
    cache = ROOT / 'data/cedar_handcrafted.npz'
    if not cache.exists():
        out = {}
        for kind in ('original', 'forgeries'):
            feats = [extract(cedar_path(kind, u, i))[0]
                     for u in range(1, N_USERS + 1) for i in range(1, N_SIGS + 1)]
            for g, name in enumerate(GROUPS):
                out[f'{kind}_{name}'] = np.stack([f[g] for f in feats])
        np.savez(cache, **out)
    d = np.load(cache)
    return {g: (d[f'original_{g}'].reshape(N_USERS, N_SIGS, -1), d[f'forgeries_{g}'].reshape(N_USERS, N_SIGS, -1))
            for g in GROUPS}


def pooled_sigma(genuine, floor):
    """人內變異：每個人自己簽名之間的標準差，再對所有人取平均。
    floor：HOG/LBP 大部分格子是空白背景，σ = 0 會除以 0，下限設為非零 σ 的中位數。
    幾何特徵不設：7 個特徵單位各不相同，共用下限會把小單位的特徵（墨跡密度）壓成 0σ"""
    sigma = np.sqrt((genuine.std(axis=1) ** 2).mean(axis=0))
    return np.maximum(sigma, np.median(sigma[sigma > 0])) if floor else sigma


def z_scores(query, prototype, sigma):
    return (query - prototype) / sigma


def group_distance(z):
    return np.sqrt((z ** 2).mean(axis=-1))  # 該組所有維度的均方根偏離，單位是 σ


def total_distance(zs):
    return np.mean([group_distance(z) for z in zs], axis=0)  # 三組等權平均，避免 HOG 2970 維壓過幾何 7 維


def calibrate():
    """用 CEDAR 校準：σ 取後 25 人，在前 30 人（每人 3 張參考）上評估各方法。
    回傳 sigmas, {方法: (EER 全域閾值, EER 每人閾值, 隨機偽造誤收率, 閾值)}；手工方法的閾值是距離（σ）"""
    feats = cedar_handcrafted()
    sigmas = {g: pooled_sigma(feats[g][0][30:], floor=g != '幾何') for g in GROUPS}
    signet_gen, signet_forg = cedar_features()
    rng = np.random.RandomState(0)

    # score 取負距離：eer() 預期分數愈高愈像本人
    methods = [*GROUPS, '手工合併', 'SigNet 餘弦']
    scores = {m: ([], [], []) for m in methods}
    for u in EVAL_USERS:
        idx = rng.permutation(N_SIGS)
        ref, test = idx[:N_REF], idx[N_REF:N_REF + 10]
        others = [v for v in EVAL_USERS if v != u]
        zs = {}
        for g in GROUPS:
            gen, forg = feats[g]
            proto = gen[u, ref].mean(0)
            zs[g] = [z_scores(x, proto, sigmas[g]) for x in (gen[u, test], forg[u], gen[others].reshape(-1, gen.shape[-1]))]
            for s, z in zip(scores[g], zs[g]):
                s.append(-group_distance(z))
        for k in range(3):
            scores['手工合併'][k].append(-total_distance([zs[g][k] for g in GROUPS]))
        proto = signet_gen[u, ref].mean(0)
        for s, x in zip(scores['SigNet 餘弦'], (signet_gen[u, test], signet_forg[u], signet_gen[others].reshape(-1, signet_gen.shape[-1]))):
            s.append(x @ proto / (np.linalg.norm(x, axis=-1) * np.linalg.norm(proto)))

    results = {}
    for m in methods:
        g_, sk, rd = (np.concatenate(s) for s in scores[m])
        global_eer, thr = eer(g_, sk)
        user_eer = np.mean([eer(a, b)[0] for a, b in zip(scores[m][0], scores[m][1])])
        results[m] = (global_eer, user_eer, np.mean(rd >= thr), thr if m == 'SigNet 餘弦' else -thr)
    return sigmas, results


def verify(ref_paths, queries, sigmas):
    """ref_paths：本人參考簽名；queries：[(標籤, 路徑)]。回傳 (參考平均圖, 本人平均特徵, 每張待測的結果)"""
    refs = [extract(p) for p in ref_paths]
    protos = [np.mean([r[0][g] for r in refs], axis=0) for g in range(len(GROUPS))]
    cases = []
    for label, path in queries:
        f, processed = extract(path)
        zs = [z_scores(f[g], protos[g], sigmas[name]) for g, name in enumerate(GROUPS)]
        groups = [group_distance(z) for z in zs]
        cases.append(dict(label=label, name=Path(path).name, raw=f[0], z=zs, img=processed, groups=groups,
                          distance=total_distance(zs)))
    return np.mean([r[1] for r in refs], axis=0), protos, cases


def plot_report(ref_mean_img, protos, n_ref, cases, threshold, title=None, bars=True):
    """上排：本人參考 + 每張待測的 HOG 熱區；下排（bars=True）：幾何特徵偏離。回傳 matplotlib Figure"""
    plt.rcParams.update({'font.family': 'PingFang TC', 'font.size': 10, 'axes.edgecolor': '#c3c2b7',
                         'text.color': '#0b0b0b', 'axes.labelcolor': '#52514e',
                         'xtick.color': '#898781', 'ytick.color': '#52514e'})
    heat_cmap = LinearSegmentedColormap.from_list('seq_blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab'])
    heat_max = 4.0  # 所有熱區共用色階，顏色深淺才能互相比較
    n_cols = 1 + len(cases)
    n_rows = 2 if bars else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.2 * n_rows), squeeze=False,
                             gridspec_kw={'wspace': 0.55 if bars else 0.12}, facecolor='#fcfcfb')
    extent = (0, 220, 150, 0)  # 圖和 HOG 格子共用同一個像素座標

    def ink_layer(img, color, alpha):
        rgba = np.zeros((*img.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(color)
        # processed 是反色圖（墨跡亮、背景 0）；縮圖後細筆畫只剩淡灰，按實際亮度拉滿才看得清楚
        rgba[..., 3] = np.clip(img / np.percentile(img[img > 0], 95), 0, 1) * alpha
        return rgba

    # 上排裁到同一個範圍：所有筆畫的外框外擴一格，並對齊 HOG 格線
    rows, cols = np.nonzero(np.max([ref_mean_img, *(c['img'] for c in cases)], axis=0) > 20)
    snap = lambda v, hi, d: int(np.clip((v // HOG_CELL + d) * HOG_CELL, 0, hi))  # noqa: E731
    y0, y1, x0, x1 = snap(rows.min(), 150, -1), snap(rows.max(), 150, 2), snap(cols.min(), 220, -1), snap(cols.max(), 220, 2)

    axes[0, 0].imshow(ink_layer(ref_mean_img, '#0b0b0b', 1.0), extent=extent)
    axes[0, 0].set_title(f'本人參考（{n_ref} 張平均）', loc='left', fontsize=12)
    for ax, c in zip(axes[0, 1:], cases):
        cell_dev = group_distance(c['z'][1].reshape(*HOG_GRID, 9))  # 每格 9 個方向的均方根偏離
        im = ax.imshow(cell_dev, cmap=heat_cmap, vmin=0, vmax=heat_max, extent=extent, interpolation='nearest')
        ax.imshow(ink_layer(ref_mean_img, '#898781', 0.45), extent=extent)  # 淡灰：本人平常的筆畫位置
        ax.imshow(ink_layer(c['img'], '#0b0b0b', 1.0), extent=extent)       # 黑：這張待測簽名
        d = c['distance']
        ax.set_title(f'{c["label"]}  {c["name"]}\n總距離 {d:.2f}σ（閾值 {threshold:.2f}σ）→ '
                     f'{"本人" if d <= threshold else "拒絕"}', loc='left', fontsize=12)
    for ax in axes[0]:
        ax.set_xlim(x0, x1), ax.set_ylim(y1, y0)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_facecolor('#fcfcfb')
        for spine in ax.spines.values():
            spine.set_color('#e1e0d9')
    cbar = fig.colorbar(im, ax=list(axes[0, 1:]), fraction=0.02, pad=0.02)
    cbar.set_label('HOG 格子偏離（σ）')
    cbar.outline.set_visible(False)

    if title:
        fig.suptitle(title, x=0.01, ha='left', fontsize=15)
    if not bars:
        return fig

    # 下排左：幾何特徵本人平均值 + 圖例說明
    axes[1, 0].axis('off')
    lines = ['本人平均（幾何特徵原始值）', ''] + [f'{n}：{v:.2f}' for n, v in zip(GEO_NAMES, protos[0])]
    lines += ['', '灰色背景 = 本人參考筆畫', '黑色 = 待測簽名', '藍色格子愈深 = 該處筆畫方向愈不像本人']
    axes[1, 0].text(0, 1, '\n'.join(lines), va='top', fontsize=11, color='#52514e', linespacing=1.6)

    z_lim = max(4.0, max(np.abs(c['z'][0]).max() for c in cases) * 1.15)
    y = np.arange(len(GEO_NAMES))[::-1]
    for ax, c in zip(axes[1, 1:], cases):
        ax.axvspan(-2, 2, color='#f0efec', zorder=0)
        ax.axvline(0, color='#c3c2b7', lw=1, zorder=1)
        ax.barh(y, c['z'][0], height=0.55, color='#2a78d6', zorder=2)
        for yi, zv, raw in zip(y, c['z'][0], c['raw']):  # 數值放在圖外右側一欄，長條再長也不會撞字
            ax.text(1.02, yi, f'{raw:.0f}' if abs(raw) >= 100 else f'{raw:.2f}', transform=ax.get_yaxis_transform(),
                    va='center', ha='left', fontsize=9, color='#52514e')
            ax.text(1.20, yi, f'{zv:+.1f}σ', transform=ax.get_yaxis_transform(), va='center', ha='left',
                    fontsize=9, color='#0b0b0b' if abs(zv) > 2 else '#898781',
                    fontweight='semibold' if abs(zv) > 2 else 'normal')
        ax.set_yticks(y, GEO_NAMES if ax is axes[1, 1] else [])  # 特徵順序相同，只在第一格標名稱
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(-z_lim, z_lim)
        ax.set_xlabel('偏離本人平均（σ）；灰色帶 = ±2σ 常見範圍')
        ax.set_title('幾何 {:.2f}σ · HOG {:.2f}σ · LBP {:.2f}σ'.format(*c['groups']), loc='left', fontsize=11,
                     color='#52514e')
        ax.set_facecolor('#fcfcfb')
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
    return fig


if __name__ == '__main__':
    sigmas, results = calibrate()
    print(f'CEDAR 前 30 人，每人 {N_REF} 張參考簽名；同一份切分比較手工特徵和 SigNet\n')
    print(f'{"方法":<14}{"EER 全域閾值":>12}{"EER 每人閾值":>12}{"隨機偽造誤收率":>14}')
    for m, (g_eer, u_eer, far, _) in results.items():
        print(f'{m:<14}{g_eer:>12.1%}{u_eer:>12.1%}{far:>14.1%}')

    # ---- 單一案例：第 1 人，前 3 張註冊，驗證一張真簽名和一張偽造（跟 Grad-CAM 示範同一組）----
    user = 1
    threshold = results['手工合併'][3]
    ref_mean_img, protos, cases = verify(
        [cedar_path('original', user, i) for i in range(1, N_REF + 1)],
        [('真簽名', cedar_path('original', user, 20)), ('偽造', cedar_path('forgeries', user, 5))], sigmas)

    print(f'\n範例：第 {user} 人，用 original_{user}_1~{N_REF} 註冊；閾值 {threshold:.2f}σ（上表「手工合併」的 EER 點）\n')
    print(f'{"幾何特徵":<14}{"本人平均":>10}' + ''.join(f'{c["label"]:>18}' for c in cases))
    for k, name in enumerate(GEO_NAMES):
        print(f'{name:<14}{protos[0][k]:>10.2f}' + ''.join(f'{c["raw"][k]:>10.2f} ({c["z"][0][k]:+5.1f}σ)' for c in cases))
    print()
    for g, name in enumerate(GROUPS):
        print(f'{name + " 組距離":<14}{"":>10}' + ''.join(f'{c["groups"][g]:>17.2f}σ' for c in cases))
    print(f'{"總距離":<14}{"":>10}' + ''.join(f'{c["distance"]:>17.2f}σ' for c in cases))
    print(f'{"判定":<14}{"":>10}' + ''.join(f'{"本人" if c["distance"] <= threshold else "拒絕":>18}' for c in cases))

    fig = plot_report(ref_mean_img, protos, N_REF, cases, threshold, f'手工特徵驗證：CEDAR 第 {user} 人')
    fig.savefig(ROOT / 'handcrafted_report.png', dpi=130, bbox_inches='tight', facecolor='#fcfcfb')
    print('\n圖已存：handcrafted_report.png')
