"""訓練鑑識版的模型：差值向量 → Elastic Net → 邏輯迴歸校準 → likelihood ratio

依 docs/SPEC.md 第 6 節。資料切分（CEDAR 55 位寫者，同一位寫者的配對不跨切分）：

| 用途 | 寫者 | 說明 |
|---|---|---|
| 訓練 | 31–48（18 人） | 學 Elastic Net 的權重 |
| 校準 | 49–55（7 人） | 把分數映射成 LR。必須用模型沒見過的寫者，否則 LR 會系統性偏大 |
| 評估 | 1–30（30 人） | 完全不參與訓練與校準，validate.py 用 |

    .venv/bin/python train.py        # 輸出 models/forensic.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupKFold

import features as F

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / 'models/forensic.json'

TRAIN_USERS = range(30, 48)      # 0-based：第 31～48 人
CALIB_USERS = range(48, 55)      # 第 49～55 人
EVAL_USERS = range(0, 30)        # 第 1～30 人，validate.py 用
SIGMA_USERS = range(30, 55)      # σ 取訓練 + 校準的寫者，評估集完全不碰

# 每位訓練寫者取幾組配對（熟練偽造:隨機 ≈ 2:1，鑑識案件幾乎都是刻意模仿）
PAIRS_PER_USER = {'genuine': 120, 'skilled': 120, 'random': 60}
ELASTIC_L1_RATIOS = (0.15, 0.5, 0.85)
ELASTIC_ALPHAS = (1e-5, 1e-4, 1e-3)
SEED = 0


def build_sigmas(feats):
    """每一族一組 σ，只用 SIGMA_USERS 的真簽名估"""
    return {fam: F.pooled_sigma(feats[fam]['original'][list(SIGMA_USERS)], floor=fam in ('HOG', 'LBP'))
            for fam in F.FAMILIES}


def diff_vector(a, b, sigmas, sift=None):
    """差值向量：各維取絕對差再除以 σ；SIFT 是成對算出來的，直接接在後面（不相減）

    a、b 是 {族: 特徵向量} 的字典。回傳一維陣列，順序固定為 F.ALL_FAMILIES。"""
    parts = [np.abs(a[fam] - b[fam]) / sigmas[fam] for fam in F.FAMILIES]
    parts.append(sift if sift is not None else np.zeros(F.DIMS['SIFT']))
    return np.concatenate(parts)


def family_slices():
    """每一族在差值向量裡的位置，供拆解貢獻用"""
    out, start = {}, 0
    for fam in F.ALL_FAMILIES:
        out[fam] = (start, start + F.DIMS[fam])
        start += F.DIMS[fam]
    return out


def sample_pairs(feats, sift, sigmas, users, rng):
    """回傳 (X, y, groups)。y=1 同一人，y=0 不同人；groups 是寫者編號，給分組交叉驗證用"""
    def vec(kind, u, i):
        return {fam: feats[fam][kind][u, i] for fam in F.FAMILIES}

    def sift_of(kind, u, i):
        return sift[(kind, u + 1, i + 1)]                     # 快取的 user/i 是 1 起算

    X, y, groups = [], [], []
    users = list(users)
    for u in users:
        plan = [('genuine', PAIRS_PER_USER['genuine']), ('skilled', PAIRS_PER_USER['skilled']),
                ('random', PAIRS_PER_USER['random'])]
        for kind, n in plan:
            for _ in range(n):
                i = rng.integers(F.N_SIGS)
                if kind == 'genuine':
                    j = rng.integers(F.N_SIGS - 1)
                    j = j + 1 if j >= i else j                    # 不跟自己配
                    b_kind, b_user, b_i, label = 'original', u, j, 1
                elif kind == 'skilled':
                    b_kind, b_user, b_i, label = 'forgeries', u, rng.integers(F.N_SIGS), 0
                else:
                    other = rng.choice([v for v in users if v != u])
                    b_kind, b_user, b_i, label = 'original', other, rng.integers(F.N_SIGS), 0
                X.append(diff_vector(vec('original', u, i), vec(b_kind, b_user, b_i), sigmas,
                                     F.sift_pair(sift_of('original', u, i), sift_of(b_kind, b_user, b_i))))
                y.append(label)
                groups.append(u)
    return np.array(X), np.array(y), np.array(groups)


def cached_pairs(feats, sift, sigmas, users, tag):
    """配對取樣的結果存成快取：SIFT 成對比較每組要 42 ms，5400 組就要近 4 分鐘，
    train.py 和 validate.py 共用同一份，不必各算一次。

    config 記錄取樣設定；設定或 σ 一變就重算。"""
    cache = ROOT / f'data/pairs_{tag}.npz'
    config = json.dumps({'users': list(users), 'plan': PAIRS_PER_USER, 'seed': SEED,
                         'sigma_sum': float(sum(v.sum() for v in sigmas.values()))}, sort_keys=True)
    if cache.exists():
        d = np.load(cache)
        if str(d['config']) == config:
            return d['X'], d['y'], d['groups']
        print(f'  （{tag} 配對快取的設定已變，重新取樣）', flush=True)
    X, y, groups = sample_pairs(feats, sift, sigmas, users, np.random.default_rng(SEED))
    np.savez(cache, X=X, y=y, groups=groups, config=config)
    return X, y, groups


def fit_elastic_net(X, y, groups):
    """以寫者為單位分組交叉驗證挑超參數。隨機切分會讓同一寫者的配對落在兩邊，高估效果。

    用 SGDClassifier(loss='log_loss', penalty='elasticnet')：它同時支援 L1+L2 混合與機率輸出，
    sklearn 的 LogisticRegression 只有 saga solver 支援 elasticnet，在這個維度下慢得多。"""
    best = None
    cv = GroupKFold(n_splits=3)
    for l1_ratio in ELASTIC_L1_RATIOS:
        for alpha in ELASTIC_ALPHAS:
            scores = []
            for tr, va in cv.split(X, y, groups):
                m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio,
                                  max_iter=3000, tol=1e-4, random_state=SEED, class_weight='balanced')
                m.fit(X[tr], y[tr])
                s = m.decision_function(X[va])
                scores.append(1 - F.eer(s[y[va] == 1], s[y[va] == 0])[0])   # 用 1-EER 當分數
            mean = float(np.mean(scores))
            print(f'  alpha={alpha:<8g} l1_ratio={l1_ratio:<5} 交叉驗證 1-EER={mean:.4f}', flush=True)
            if best is None or mean > best[0]:
                best = (mean, alpha, l1_ratio)

    _, alpha, l1_ratio = best
    model = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio,
                          max_iter=5000, tol=1e-5, random_state=SEED, class_weight='balanced')
    model.fit(X, y)
    print(f'  選定 alpha={alpha:g} l1_ratio={l1_ratio}；非零權重 '
          f'{int((model.coef_[0] != 0).sum())}/{X.shape[1]}')
    return model, alpha, l1_ratio


def fit_calibration(scores, labels):
    """分數 → 對數 LR。單變數邏輯迴歸，係數 (a, b) 讓 log LR = a * score + b"""
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(scores.reshape(-1, 1), labels)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


if __name__ == '__main__':
    print('載入特徵快取……', flush=True)
    feats, sift = F.cedar_features(), F.cedar_sift()
    sigmas = build_sigmas(feats)

    print(f'\n取樣訓練配對（{len(list(TRAIN_USERS))} 位寫者，每人 '
          f'{sum(PAIRS_PER_USER.values())} 組）……')
    Xtr, ytr, gtr = cached_pairs(feats, sift, sigmas, TRAIN_USERS, 'train')
    print(f'  {Xtr.shape[0]} 組配對 × {Xtr.shape[1]} 維（同一人 {int(ytr.sum())}，不同人 {int((1 - ytr).sum())}）', flush=True)

    print('\n分組交叉驗證挑超參數……', flush=True)
    model, alpha, l1_ratio = fit_elastic_net(Xtr, ytr, gtr)

    # 同源配對的典型差異，當作貢獻拆解的基準。
    # 分數 = Σ 權重ᵢ×(xᵢ − 基準ᵢ) + (截距 + Σ 權重ᵢ×基準ᵢ)，是精確的代數重寫。
    # 沒有基準的話，貢獻 = 權重 × 絕對差，符號全由權重決定，長條會全部朝同一邊。
    baseline = Xtr[ytr == 1].mean(axis=0)

    print(f'\n取樣校準配對（{len(list(CALIB_USERS))} 位寫者，模型沒見過）……', flush=True)
    Xca, yca, _ = cached_pairs(feats, sift, sigmas, CALIB_USERS, 'calib')
    cal_a, cal_b = fit_calibration(model.decision_function(Xca), yca)
    print(f'  log LR = {cal_a:.4f} × 分數 + {cal_b:.4f}', flush=True)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({
        'families': list(F.ALL_FAMILIES),
        'dims': {k: int(v) for k, v in F.DIMS.items()},
        'slices': {k: list(v) for k, v in family_slices().items()},
        'sigmas': {fam: sigmas[fam].tolist() for fam in F.FAMILIES},
        'coef': model.coef_[0].tolist(),
        'intercept': float(model.intercept_[0]),
        'baseline_diff': baseline.tolist(),   # 同源配對的典型差異，拆解貢獻用
        'calibration': {'a': cal_a, 'b': cal_b},
        'hyperparams': {'alpha': alpha, 'l1_ratio': l1_ratio},
        'train_users': list(TRAIN_USERS), 'calib_users': list(CALIB_USERS), 'eval_users': list(EVAL_USERS),
        'pairs_per_user': PAIRS_PER_USER,
        'dataset': 'CEDAR（西方拉丁字母簽名）—— 原型數字，不可用於實際案件',
    }, ensure_ascii=False, indent=1))
    print(f'\n已存 {MODEL_PATH.relative_to(ROOT)}', flush=True)
