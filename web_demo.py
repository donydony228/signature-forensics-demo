"""模擬「同一份文件裡取樣一批墨水區塊的反射率光譜，用 t-SNE 檢查是否混用了不只一種墨水」。

流程比照 Forensic Sci Int 311(2020)110194：光譜直接送進 PCA / t-SNE，再用 KMeans 分群。
手上沒有 HySpex 拍的真實光譜，所以用仿論文 Fig. 4 形狀的模擬光譜代替（400–1000 nm、186 個波段）。
拿到真實光譜後，只要把 DOC_X 換成「每列一個樣本、每欄一個波長」的反射率矩陣，後面不用改。

這是論文原本的用法，不是「拿一張新圖跟已知嫌疑人比對」：
一份文件裡的樣本全部一起丟進 PCA/t-SNE，不用任何身分標籤，
純粹看樣本會不會自己分成不只一群——多一群，就是有混用不同墨水/筆跡的訊號。
demo 文件裡混了幾個「疑似竄改」的區塊，真實來源只在動畫跑完後才揭曉，
純粹是這個 demo 自己驗證用，真實案件不會有這個答案。

前後端分離：這支只負責運算（特徵/PCA/t-SNE/KMeans 全部 Python），
靜態頁面在 index.html，用 fetch 打 /api/analyze-document 拿 JSON 座標，
前端用 canvas 畫，沒有手刻的 JS 演算法。

t-SNE 沒有 out-of-sample transform，這裡也用不到——這次沒有「新樣本」，
整批文件樣本本來就是一起降維的，跟論文原本的做法一致。

t-SNE 動畫的做法：sklearn 的 TSNE 不提供「每一步」的 callback，
但給定同樣的 random_state/init，優化過程是確定性的，所以用遞增的 max_iter
重跑好幾次，就等於拿到同一條收斂軌跡上不同時間點的快照（真的中間結果，不是插值動畫）。

    .venv/bin/python web_demo.py            # 啟動網頁，http://127.0.0.1:5000
    .venv/bin/python web_demo.py --check     # 不開網頁，跑一次 pipeline 自我檢查
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_from_directory
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.metrics import completeness_score, homogeneity_score, normalized_mutual_info_score, silhouette_score

WAVELENGTHS = np.linspace(400, 1000, 186)  # nm，對應 HySpex VNIR-1800 的 186 個波段

# 模擬一份文件：大部分區塊出自同一來源，少數混入另一來源（竄改/插入）。
DOC_N_NORMAL = 28
DOC_N_SUSPICIOUS = 7
DOC_SEED = 0

# 快照間距不是均勻的：早期（250~350）分群變動最大，量過（mean movement）確認過，
# 後段（350~1000）已經大致收斂，動得很少，切太多段沒意義，所以前段擠、後段拉開。
TSNE_SNAPSHOT_ITERS = (250, 265, 280, 300, 325, 350, 450, 600, 800, 1000)  # sklearn 要求 max_iter >= 250


# 兩種藍墨水：可見光段都在 450 nm 附近有個反射峰，差在近紅外「往上翻」的吸收邊緣位置（705 vs 740 nm）。
# 差距刻意做小（35 nm）：太好分的話每種降維方法都滿分，方法比較就沒意義。
INKS = {0: {"edge": 705, "peak": 0.42}, 1: {"edge": 740, "peak": 0.36}}
EDGE_JITTER_NM = 4   # 同一支筆不同位置，吸收邊緣會飄
PAPER_MIX_MAX = 0.15  # 墨水筆劃很細，像素裡會混到紙張，最多 15%
THICKNESS_SD = 0.05   # 墨水厚薄造成的整體亮度變化
PAPER_REFLECTANCE = 0.86
NOISE_SD = 0.012


def make_ink_spectrum(source_id: int, rng: np.random.Generator) -> np.ndarray:
    """一個墨水像素的反射率光譜（0–1），形狀仿論文 Fig. 4 的藍筆：450 nm 反射峰 + 近紅外吸收邊緣往上翻。"""
    ink = INKS[source_id]
    edge = ink["edge"] + rng.normal(0, EDGE_JITTER_NM)
    thickness = rng.normal(1, THICKNESS_SD)
    paper = rng.uniform(0, PAPER_MIX_MAX)
    wl = WAVELENGTHS
    pure = 0.08 + ink["peak"] * np.exp(-((wl - 450) / 38) ** 2) + 0.74 / (1 + np.exp(-(wl - edge) / 16))
    mixed = (1 - paper) * pure * thickness + paper * PAPER_REFLECTANCE
    return mixed + rng.normal(0, NOISE_SD, wl.size)


def build_document_sample():
    """true_source：0 = 文件原本的來源，1 = 疑似竄改插入的來源。只給 demo 事後驗證用。"""
    rng = np.random.default_rng(DOC_SEED)
    sources = np.array([0] * DOC_N_NORMAL + [1] * DOC_N_SUSPICIOUS)
    np.random.default_rng(DOC_SEED + 1).shuffle(sources)  # 打散順序，不要正常/竄改連續排列

    spectra = np.array([make_ink_spectrum(int(s), rng) for s in sources])
    return spectra, sources


DOC_X, DOC_TRUE_SOURCE = build_document_sample()


K_CANDIDATES = range(2, 7)
# ponytail: 「高峰明顯」的門檻只用一組混用、一組單一來源的合成資料定過（0.08 vs 0.01），
# 真實使用前要用實驗室已知答案的樣本重新校正。
CLEAR_PEAK_MARGIN = 0.05


def choose_k(X: np.ndarray) -> dict:
    """不看答案決定群數：在 PCA 空間（不是 t-SNE 圖上，t-SNE 會把資料拉成假的團塊）用輪廓係數試 k=2~6。
    k=1（沒有混用）目前無法可靠地自動判定（gap statistic、GMM+BIC 都在單一來源資料上誤判過），
    所以只回報曲線有沒有明顯高峰，由人判讀。"""
    Z = PCA(n_components=min(10, len(X) - 1), random_state=0).fit_transform(X)
    scores = {k: float(silhouette_score(Z, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)))
              for k in K_CANDIDATES}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, runner_up = ranked[0], ranked[1]
    clear = best != max(K_CANDIDATES) and scores[best] - scores[runner_up] >= CLEAR_PEAK_MARGIN
    return {"k": best, "silhouette": scores, "clear_peak": clear, "pca_dims": Z.shape[1]}


def cluster_metrics(embedding: np.ndarray, y: np.ndarray, k: int):
    pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embedding)
    metrics = {
        "SI": float(silhouette_score(embedding, pred)),
        "NMI": float(normalized_mutual_info_score(y, pred)),
        "HI": float(homogeneity_score(y, pred)),
        "CI": float(completeness_score(y, pred)),
    }
    return metrics, pred


# 動畫用 random 初始化單跑一次；種子要讓最後結果跟 cluster_metrics（n_init=10）一致。
# 光譜資料在 t-SNE 上分得很開，試過的種子最多 2 步就收斂；刻意挑爛起點（資料中心附近）
# 雖然步數變多，但會卡在錯誤的分法、跟指標表格矛盾，所以不採用。
KMEANS_ANIM_SEED = 2


def run_kmeans_trajectory(embedding: np.ndarray, k: int):
    full = KMeans(n_clusters=k, n_init=1, init="random", max_iter=100, random_state=KMEANS_ANIM_SEED).fit(embedding)
    frames = []
    for m in range(1, full.n_iter_ + 1):
        km = KMeans(n_clusters=k, n_init=1, init="random", max_iter=m, random_state=KMEANS_ANIM_SEED).fit(embedding)
        frames.append({"labels": km.labels_.tolist(), "centers": km.cluster_centers_.tolist()})
    return frames


def run_tsne_trajectory(X: np.ndarray) -> list[np.ndarray]:
    perplexity = min(30, len(X) // 3)
    return [
        TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca", max_iter=n).fit_transform(X)
        for n in TSNE_SNAPSHOT_ITERS
    ]


def run_pipeline():
    pca_2d = PCA(n_components=2, random_state=0).fit_transform(DOC_X)
    tsne_frames = run_tsne_trajectory(DOC_X)
    tsne_2d = tsne_frames[-1]

    k_selection = choose_k(DOC_X)
    k = k_selection["k"]
    pca_metrics, _ = cluster_metrics(pca_2d, DOC_TRUE_SOURCE, k)
    tsne_metrics, tsne_kmeans_labels = cluster_metrics(tsne_2d, DOC_TRUE_SOURCE, k)
    kmeans_frames = run_kmeans_trajectory(tsne_2d, k)

    return tsne_frames, pca_metrics, tsne_metrics, tsne_kmeans_labels, kmeans_frames, k_selection


# LDA 沒放：它是監督式方法，要先知道真實來源才能算，等於偷看答案。
# UMAP 沒放：需要另外安裝 umap-learn，確定要比較它時再加。
DR_METHODS = [
    ("PCA", "線性", lambda n: PCA(n_components=2, random_state=0)),
    ("ICA", "線性", lambda n: FastICA(n_components=2, random_state=0, max_iter=1000)),
    ("MDS", "非線性", lambda n: MDS(n_components=2, random_state=0)),
    ("Isomap", "非線性", lambda n: Isomap(n_components=2)),
    ("Spectral", "非線性", lambda n: SpectralEmbedding(n_components=2, random_state=0)),
    ("t-SNE", "非線性", lambda n: TSNE(n_components=2, perplexity=min(30, n // 3), random_state=0, init="pca")),
]


def compare_methods(X: np.ndarray, y: np.ndarray):
    """每種方法都降到 2 維再用 KMeans 分成 choose_k 判定的群數。排名只看 SI：SI 不需要真實來源，真實案件也算得出來；
    NMI/HI/CI 需要答案，只拿來在 demo 裡驗證自動挑的對不對。"""
    k = choose_k(X)["k"]
    results = []
    for name, kind, make in DR_METHODS:
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # MDS/Isomap/Spectral 在小資料上會噴不影響結果的提醒
            embedding = make(len(X)).fit_transform(X)
        seconds = time.perf_counter() - start
        metrics, labels = cluster_metrics(embedding, y, k)
        results.append({
            "name": name, "kind": kind, "seconds": seconds,
            "points": embedding.tolist(), "kmeans_labels": labels.tolist(), "metrics": metrics,
        })
    best = max(range(len(results)), key=lambda i: results[i]["metrics"]["SI"])
    return results, best, k


app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.get("/compare")
def compare_page():
    return send_from_directory(Path(__file__).parent, "compare.html")


@app.get("/api/analyze-document")
def analyze_document():
    tsne_frames, pca_metrics, tsne_metrics, tsne_kmeans_labels, kmeans_frames, k_selection = run_pipeline()

    return jsonify(
        frames=[frame.tolist() for frame in tsne_frames],
        iters=list(TSNE_SNAPSHOT_ITERS),
        true_source=DOC_TRUE_SOURCE.tolist(),
        wavelengths=WAVELENGTHS.round(1).tolist(),
        spectra=DOC_X.round(4).tolist(),
        tsne_kmeans_labels=tsne_kmeans_labels.tolist(),
        kmeans_frames=kmeans_frames,
        k_selection=k_selection,
        pca_metrics=pca_metrics,
        tsne_metrics=tsne_metrics,
    )


@app.get("/api/compare-methods")
def compare_methods_api():
    results, best, k = compare_methods(DOC_X, DOC_TRUE_SOURCE)
    return jsonify(methods=results, best=best, k=k, true_source=DOC_TRUE_SOURCE.tolist())


def check():
    tsne_frames, pca_metrics, tsne_metrics, tsne_kmeans_labels, kmeans_frames, k_selection = run_pipeline()

    assert k_selection["k"] == 2 and k_selection["clear_peak"], f"混用文件應判定 k=2 且高峰明顯：{k_selection}"
    single_rng = np.random.default_rng(5)
    single_doc = np.array([make_ink_spectrum(0, single_rng) for _ in range(len(DOC_X))])
    assert not choose_k(single_doc)["clear_peak"], "單一來源文件不該被標成群聚結構明顯"

    k = k_selection["k"]
    assert len(tsne_frames) == len(TSNE_SNAPSHOT_ITERS)
    assert all(frame.shape == (len(DOC_X), 2) for frame in tsne_frames)
    assert tsne_kmeans_labels.shape == (len(DOC_X),)
    assert all(0 <= v <= 1 for v in {**pca_metrics, **tsne_metrics}.values())
    assert tsne_metrics["NMI"] > 0.3, "t-SNE 完全沒分出竄改區塊，資料生成或流程可能壞了"
    assert len(kmeans_frames) >= 1
    assert all(len(f["labels"]) == len(DOC_X) and len(f["centers"]) == k for f in kmeans_frames)
    assert normalized_mutual_info_score(kmeans_frames[-1]["labels"], tsne_kmeans_labels) == 1.0, \
        "KMeans 動畫最後一幀跟表格用的分群結果對不起來"
    results, best, _ = compare_methods(DOC_X, DOC_TRUE_SOURCE)
    assert [r["name"] for r in results] == [m[0] for m in DR_METHODS]
    assert all(len(r["points"]) == len(DOC_X) for r in results)
    assert results[best]["metrics"]["SI"] == max(r["metrics"]["SI"] for r in results)
    assert len({round(r["metrics"]["NMI"], 2) for r in results}) > 1, "每種方法分數都一樣，比較沒有意義，資料可能太好分"

    print("self-check ok:", {"PCA": pca_metrics, "t-SNE": tsne_metrics, "kmeans_iters": len(kmeans_frames)})
    for i, r in enumerate(results):
        print(f"  {'*' if i == best else ' '} {r['name']:9s} SI {r['metrics']['SI']:.2f}  NMI {r['metrics']['NMI']:.2f}  {r['seconds']:.3f}s")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        app.run(debug=True)
