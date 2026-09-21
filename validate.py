"""驗證鑑識版模型：Cllr、Tippett 圖、逐族消融表

依 docs/SPEC.md 第 7 節。每個模型版本跑一次，不是每個案件跑一次。

為什麼用 Cllr 而不是 EER：EER 只看分數排序對不對，Cllr 同時罰「方向錯」和「過度自信」，
是鑑識領域評估 LR 系統的標準指標。愈低愈好，1.0 等於「完全沒有資訊量」。

熟練偽造與隨機不同人一律分開報：兩者混算會被容易分辨的隨機組把數字拉漂亮，
掩蓋熟練偽造的真實難度，而鑑識案件面對的幾乎都是前者。

    .venv/bin/python validate.py            # 完整驗證（含消融）
    .venv/bin/python validate.py --no-ablation
    .venv/bin/python validate.py --subsets  # 另外試幾種族組合（消融是一次拿一族，這個看組合效果）
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

import features as F
import train as T

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
REPORT = ROOT / 'validation_report.png'
N_REF = 3                    # 每位評估寫者的已知樣本張數
N_TEST_GENUINE = 10          # 每人拿幾張真簽名當同源試驗
N_RANDOM_PER_USER = 2        # 隨機不同人：從其他每位寫者各取幾張


def cllr(lr_same, lr_diff):
    """對數 LR 成本。1.0 = 沒有資訊量；愈低愈好。方向錯或過度自信都會被罰"""
    if len(lr_same) == 0 or len(lr_diff) == 0:
        return float('nan')
    return 0.5 * (np.mean(np.log2(1 + 1 / np.maximum(lr_same, 1e-12)))
                  + np.mean(np.log2(1 + np.maximum(lr_diff, 0))))


def build_eval_trials(feats, sift, sigmas, rng):
    """每個試驗 = 一張待鑑簽名對上 N_REF 張已知樣本。

    模型是線性的，所以「各參考樣本的分數取平均」等於「差值向量取平均後算一次分數」，
    這裡存平均後的差值向量，評估與消融都能重複使用。

    回傳 {同源/熟練偽造/隨機: 差值向量矩陣}"""
    def vec(kind, u, i):
        return {fam: feats[fam][kind][u, i] for fam in F.FAMILIES}

    trials = {'同源': [], '熟練偽造': [], '隨機不同人': []}
    users = list(T.EVAL_USERS)
    for u in users:
        idx = rng.permutation(F.N_SIGS)
        refs, tests = idx[:N_REF], idx[N_REF:N_REF + N_TEST_GENUINE]
        ref_data = [(vec('original', u, r), sift[('original', u + 1, r + 1)]) for r in refs]

        def mean_diff(kind, uu, ii):
            q, q_sift = vec(kind, uu, ii), sift[(kind, uu + 1, ii + 1)]
            # sift_pair 的第一個參數與訓練時一致（訓練時固定是本人的真簽名）
            return np.mean([T.diff_vector(q, rv, sigmas, F.sift_pair(rs, q_sift))
                            for rv, rs in ref_data], axis=0)

        for i in tests:
            trials['同源'].append(mean_diff('original', u, i))
        for i in range(F.N_SIGS):
            trials['熟練偽造'].append(mean_diff('forgeries', u, i))
        for other in users:
            if other == u:
                continue
            for i in rng.choice(F.N_SIGS, N_RANDOM_PER_USER, replace=False):
                trials['隨機不同人'].append(mean_diff('original', other, i))
    return {k: np.array(v) for k, v in trials.items()}


def cached_trials(feats, sift, sigmas):
    """評估試驗的差值向量存成快取：8280 次 SIFT 成對比較要 6 分鐘，
    換模型或跑組合實驗時不必重算（試驗只取決於 σ 與取樣設定，與權重無關）"""
    cache = ROOT / 'data/eval_trials.npz'
    config = json.dumps({'users': list(T.EVAL_USERS), 'n_ref': N_REF, 'n_test': N_TEST_GENUINE,
                         'n_random': N_RANDOM_PER_USER, 'seed': T.SEED,
                         'sigma_sum': float(sum(v.sum() for v in sigmas.values()))}, sort_keys=True)
    if cache.exists():
        d = np.load(cache)
        if str(d['config']) == config:
            return {k: d[k] for k in ('同源', '熟練偽造', '隨機不同人')}
        print('  （評估試驗快取的設定已變，重新建立）', flush=True)
    trials = build_eval_trials(feats, sift, sigmas, np.random.default_rng(T.SEED))
    np.savez(cache, config=config, **trials)
    return trials


def score_and_lr(X, coef, intercept, calib):
    """差值向量 → 原始分數 → likelihood ratio。指數要夾住，否則大分數會溢位成 inf，Cllr 跟著變 inf"""
    scores = X @ coef + intercept
    return scores, np.exp(np.clip(calib['a'] * scores + calib['b'], -30, 30))


def evaluate(trials, coef, intercept, calib):
    """回傳 {組合: {cllr, eer, lr_same, lr_diff}}"""
    lrs = {k: score_and_lr(X, coef, intercept, calib)[1] for k, X in trials.items()}
    scores = {k: score_and_lr(X, coef, intercept, calib)[0] for k, X in trials.items()}
    out = {}
    for name in ('熟練偽造', '隨機不同人'):
        out[name] = {
            'cllr': cllr(lrs['同源'], lrs[name]),
            'eer': F.eer(scores['同源'], scores[name])[0],
            'lr_same': lrs['同源'], 'lr_diff': lrs[name],
        }
    return out


def refit(Xtr, ytr, gtr, Xca, yca, keep, hyper):
    """用保留的欄位重訓 + 重新校準（消融用）。超參數沿用正式模型，不重跑交叉驗證"""
    from sklearn.linear_model import SGDClassifier
    m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=hyper['alpha'],
                      l1_ratio=hyper['l1_ratio'], max_iter=5000, tol=1e-5,
                      random_state=T.SEED, class_weight='balanced')
    m.fit(Xtr[:, keep], ytr)
    a, b = T.fit_calibration(m.decision_function(Xca[:, keep]), yca)
    return m.coef_[0], float(m.intercept_[0]), {'a': a, 'b': b}


def tippett_plot(ax, result, title):
    """Tippett 圖：橫軸是門檻，縱軸是「LR 超過該門檻的比例」。
    同源曲線應該多落在右側（LR 大），異源曲線應該多落在左側。"""
    grid = np.linspace(-6, 6, 400)
    for key, label, color in (('lr_same', '同源（同一人）', '#2a78d6'),
                              ('lr_diff', '異源（不同人）', '#d03b3b')):
        logs = np.log10(np.maximum(result[key], 1e-12))
        ax.plot(grid, [(logs >= g).mean() for g in grid], color=color, lw=2, label=label)
    ax.axvline(0, color='#c3c2b7', lw=1)
    ax.set_xlabel('門檻（log₁₀ LR）')
    ax.set_ylabel('LR 超過門檻的比例')
    ax.set_title(f'{title}\nCllr {result["cllr"]:.3f}　EER {result["eer"]:.1%}', loc='left', fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, frameon=False)
    ax.set_facecolor('#fcfcfb')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


if __name__ == '__main__':
    model = json.loads((ROOT / 'models/forensic.json').read_text())
    coef, intercept, calib = np.array(model['coef']), model['intercept'], model['calibration']
    slices = {k: tuple(v) for k, v in model['slices'].items()}

    print('載入特徵快取……', flush=True)
    feats, sift = F.cedar_features(), F.cedar_sift()
    sigmas = {fam: np.array(v) for fam, v in model['sigmas'].items()}

    print(f'建立評估試驗（{len(list(T.EVAL_USERS))} 位寫者，每人 {N_REF} 張已知樣本）……', flush=True)
    trials = cached_trials(feats, sift, sigmas)
    for k, v in trials.items():
        print(f'  {k:<8}{len(v):>6} 個試驗', flush=True)

    results = evaluate(trials, coef, intercept, calib)
    print(f'\n{"組合":<12}{"Cllr":>8}{"EER":>10}', flush=True)
    for name, r in results.items():
        print(f'{name:<12}{r["cllr"]:>8.3f}{r["eer"]:>10.1%}', flush=True)

    ablation = {}
    if '--no-ablation' not in sys.argv:
        print('\n逐族消融（拿掉該族後重訓 + 重新校準）……', flush=True)
        Xtr, ytr, gtr = T.cached_pairs(feats, sift, sigmas, T.TRAIN_USERS, 'train')
        Xca, yca, _ = T.cached_pairs(feats, sift, sigmas, T.CALIB_USERS, 'calib')
        for fam in F.ALL_FAMILIES:
            lo, hi = slices[fam]
            keep = np.r_[np.arange(0, lo), np.arange(hi, Xtr.shape[1])]
            c, b0, cal = refit(Xtr, ytr, gtr, Xca, yca, keep, model['hyperparams'])
            r = evaluate({k: v[:, keep] for k, v in trials.items()}, c, b0, cal)
            ablation[fam] = {k: v['cllr'] for k, v in r.items()}
            delta = ablation[fam]['熟練偽造'] - results['熟練偽造']['cllr']
            verdict = '有貢獻' if delta > 0.01 else ('無貢獻' if delta > -0.01 else '拿掉更好')
            print(f'  拿掉 {fam:<8} Cllr(熟練偽造) {ablation[fam]["熟練偽造"]:.3f} '
                  f'（{delta:+.3f}）{verdict}')

    if '--subsets' in sys.argv:
        print('\n族組合實驗（重訓 + 重新校準，Cllr 為熟練偽造組）……', flush=True)
        Xtr, ytr, gtr = T.cached_pairs(feats, sift, sigmas, T.TRAIN_USERS, 'train')
        Xca, yca, _ = T.cached_pairs(feats, sift, sigmas, T.CALIB_USERS, 'calib')
        subsets = [
            ('全部 8 族', F.ALL_FAMILIES),
            ('不含 HOG', [f for f in F.ALL_FAMILIES if f != 'HOG']),
            ('不含 HOG、LBP', [f for f in F.ALL_FAMILIES if f not in ('HOG', 'LBP')]),
            ('不含 HOG、LBP、SIFT', [f for f in F.ALL_FAMILIES if f not in ('HOG', 'LBP', 'SIFT')]),
            ('幾何+方向+小波+GLCM', ['幾何', '方向分布', '小波', 'GLCM']),
            ('方向+小波+GLCM（無幾何）', ['方向分布', '小波', 'GLCM']),
            ('幾何+方向+GLCM', ['幾何', '方向分布', 'GLCM']),
            ('只有 GLCM', ['GLCM']),
            ('只有幾何', ['幾何']),
        ]
        print(f'{"組合":<26}{"維度":>6}{"非零":>6}{"Cllr 熟練":>10}{"Cllr 隨機":>10}{"EER 熟練":>10}', flush=True)
        for label, fams in subsets:
            keep = np.concatenate([np.arange(*slices[f]) for f in fams])
            c, b0, cal = refit(Xtr, ytr, gtr, Xca, yca, keep, model['hyperparams'])
            r = evaluate({k: v[:, keep] for k, v in trials.items()}, c, b0, cal)
            print(f'{label:<26}{len(keep):>6}{int((c != 0).sum()):>6}'
                  f'{r["熟練偽造"]["cllr"]:>10.3f}{r["隨機不同人"]["cllr"]:>10.3f}'
                  f'{r["熟練偽造"]["eer"]:>10.1%}', flush=True)

    plt.rcParams.update({'font.family': 'PingFang TC', 'font.size': 10, 'axes.edgecolor': '#c3c2b7',
                         'text.color': '#0b0b0b', 'axes.labelcolor': '#52514e',
                         'xtick.color': '#898781', 'ytick.color': '#52514e'})
    n_cols = 3 if ablation else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5.4 * n_cols, 4.4), facecolor='#fcfcfb')
    tippett_plot(axes[0], results['熟練偽造'], 'Tippett 圖：熟練偽造')
    tippett_plot(axes[1], results['隨機不同人'], 'Tippett 圖：隨機不同人')
    if ablation:
        ax = axes[2]
        fams = list(ablation)
        deltas = [ablation[f]['熟練偽造'] - results['熟練偽造']['cllr'] for f in fams]
        y = np.arange(len(fams))[::-1]
        ax.barh(y, deltas, height=0.6, color=['#2a78d6' if d > 0 else '#d03b3b' for d in deltas])
        ax.axvline(0, color='#c3c2b7', lw=1)
        ax.set_yticks(y, fams)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('拿掉該族後 Cllr 的變化（熟練偽造）')
        ax.set_title('消融：往右 = 該族有貢獻\n往左 = 拿掉反而更好', loc='left', fontsize=11)
        ax.set_facecolor('#fcfcfb')
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)

    fig.suptitle(f'鑑識版驗證　CEDAR 第 1–{len(list(T.EVAL_USERS))} 人　'
                 f'原型數字，不可用於實際案件', x=0.01, ha='left', fontsize=13)
    fig.tight_layout()
    fig.savefig(REPORT, dpi=130, bbox_inches='tight', facecolor='#fcfcfb')
    print(f'\n圖已存 {REPORT.name}', flush=True)

    # 驗證摘要寫回模型檔，鑑識版頁面會顯示
    model['validation'] = {
        'n_eval_users': len(list(T.EVAL_USERS)), 'n_ref': N_REF,
        'trials': {k: len(v) for k, v in trials.items()},
        'cllr': {k: float(v['cllr']) for k, v in results.items()},
        'eer': {k: float(v['eer']) for k, v in results.items()},
        'ablation_cllr_skilled': {k: float(v['熟練偽造']) for k, v in ablation.items()} or None,
    }
    (ROOT / 'models/forensic.json').write_text(json.dumps(model, ensure_ascii=False, indent=1))
    print('驗證摘要已寫回 models/forensic.json', flush=True)
