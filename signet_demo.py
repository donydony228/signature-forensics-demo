"""SigNet 簽名驗證示範（CEDAR 資料集）—— 度量學習版

流程：簽名圖 → 前處理 → SigNet 抽 2048 維特徵 → 和本人參考簽名的平均向量算餘弦相似度 → 分數 ≥ 閾值 = 本人

不訓練任何分類器：本人的「模型」就是他幾張參考簽名的平均特徵（prototype），新人要註冊
只要抽幾張簽名的特徵、取平均，不需要蒐集別人的簽名當負樣本，也不用重新訓練。

    .venv/bin/python signet_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import img_as_ubyte
from skimage.io import imread

from features import CANVAS, N_SIGS, N_USERS, cedar_path, eer  # 共用常數與工具

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'sigver'))
from sigver.featurelearning.models import SigNet  # noqa: E402
from sigver.preprocessing.normalize import preprocess_signature  # noqa: E402

# 注意：這支是 SigNet 對照組，前處理刻意維持原論文的做法（不做 features.load_gray 的尺寸正規化），
# 因為 data/cedar_signet.npz 是照這個流程建的。

state_dict, _, _ = torch.load(ROOT / 'sigver/models/signet.pth', weights_only=False)
signet = SigNet().eval()
signet.load_state_dict(state_dict)


def embed(paths):
    """簽名圖檔 → SigNet 特徵 (N, 2048)"""
    x = np.stack([preprocess_signature(img_as_ubyte(imread(p, as_gray=True)), CANVAS) for p in paths])
    x = torch.from_numpy(x).unsqueeze(1).float().div(255)
    with torch.no_grad():
        return torch.cat([signet(b) for b in x.split(64)]).numpy()


def cedar_features():
    """全部 2640 張抽特徵，快取到 npz。回傳 genuine, forgery，形狀都是 (55, 24, 2048)"""
    cache = ROOT / 'data/cedar_signet.npz'
    if not cache.exists():
        feats = {kind: embed([cedar_path(kind, u, i) for u in range(1, N_USERS + 1) for i in range(1, N_SIGS + 1)])
                 for kind in ('original', 'forgeries')}
        np.savez(cache, **feats)
    d = np.load(cache)
    return d['original'].reshape(N_USERS, N_SIGS, -1), d['forgeries'].reshape(N_USERS, N_SIGS, -1)


def enroll(reference_signatures):
    """註冊：本人幾張參考簽名的平均特徵（prototype）。不訓練，不需要別人的簽名。"""
    return reference_signatures.mean(axis=0)


def cosine(features, prototype):
    return features @ prototype / (np.linalg.norm(features, axis=-1) * np.linalg.norm(prototype))


if __name__ == '__main__':
    gen, forg = cedar_features()
    exp_users = range(30)  # 前 30 人做評估
    rng = np.random.RandomState(0)

    # ponytail: 只跑一次隨機切分；正式評估要跑多次取平均，這裡先看量級
    print('CEDAR，30 人 —— 用不同數量的參考簽名註冊，看少樣本下的表現\n')
    print(f'{"參考簽名張數":<12}{"EER 全域閾值":>12}{"EER 每人閾值":>12}{"隨機偽造誤收率":>14}')
    thresholds = {}
    for n_ref in (1, 3, 5, 12):
        scores_gen, scores_skilled, scores_random = [], [], []
        for u in exp_users:
            idx = rng.permutation(N_SIGS)
            ref, test = gen[u, idx[:n_ref]], gen[u, idx[n_ref:n_ref + 10]]
            others = gen[[v for v in exp_users if v != u]].reshape(-1, gen.shape[-1])
            prototype = enroll(ref)
            scores_gen.append(cosine(test, prototype))
            scores_skilled.append(cosine(forg[u], prototype))       # 同一人的熟練偽造
            scores_random.append(cosine(others, prototype))          # 其他人的真簽名
        global_eer, thr = eer(np.concatenate(scores_gen), np.concatenate(scores_skilled))
        user_eer = np.mean([eer(g, f)[0] for g, f in zip(scores_gen, scores_skilled)])
        far_random = np.mean(np.concatenate(scores_random) >= thr)
        print(f'{n_ref:<12}{global_eer:>12.1%}{user_eer:>12.1%}{far_random:>14.1%}')
        thresholds[n_ref] = thr  # 閾值要從資料算出來，不能憑感覺猜（下面的例子會示範猜錯的後果）

    # 實際使用：只用 3 張圖註冊一個新人（沒有訓練，也沒有蒐集別人的簽名），驗證新簽名
    user, n_ref = 1, 3
    prototype = enroll(embed([cedar_path('original', user, i) for i in range(1, n_ref + 1)]))
    threshold = thresholds[n_ref]  # 沿用上面同一批人估出的全域閾值；正式系統要用獨立的驗證集校準
    print(f'\n範例：只用 {n_ref} 張圖註冊 CEDAR 第 {user} 人，閾值 {threshold:.3f}')
    for kind, i in (('original', 20), ('forgeries', 5)):
        path = cedar_path(kind, user, i)
        s = cosine(embed([path]), prototype)[0]
        print(f'  {path.name:<22} 相似度 {s:.3f} → {"本人" if s >= threshold else "拒絕"}')
