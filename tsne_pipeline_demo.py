"""t-SNE vs PCA 分群 pipeline，仿造 Forensic Sci Int 311(2020)110194 的方法，
從「幫每支筆分群」換成「幫每位筆跡作者分群」。

真實用法：把 make_synthetic_writers() 換成 features.py 算出的
X (n_samples, n_features) / y (writer_id)，其餘流程不用動。

    .venv/bin/python tsne_pipeline_demo.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)

RNG = np.random.default_rng(0)


def make_synthetic_writers(n_writers=8, n_samples=30, n_features=50, similar_pair=(2, 5)):
    """模擬多位筆跡的特徵向量；similar_pair 兩位刻意做成幾乎一樣，
    對應論文裡 Pen2/Pen5 那種光譜幾乎重疊、拖累分群品質的情境（例如臨摹得很像的偽造簽名）。
    """
    centers = RNG.normal(scale=6.0, size=(n_writers, n_features))
    a, b = similar_pair
    centers[b] = centers[a] + RNG.normal(scale=0.3, size=n_features)
    X = np.vstack([centers[i] + RNG.normal(scale=1.5, size=(n_samples, n_features)) for i in range(n_writers)])
    y = np.repeat(np.arange(n_writers), n_samples)
    return X, y


def evaluate(embedding, y, n_clusters):
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(embedding)
    return {
        "SI": silhouette_score(embedding, pred),
        "NMI": normalized_mutual_info_score(y, pred),
        "HI": homogeneity_score(y, pred),
        "CI": completeness_score(y, pred),
    }


def run_pipeline(X, y):
    n_clusters = len(np.unique(y))
    pca_2d = PCA(n_components=2, random_state=0).fit_transform(X)
    tsne_2d = TSNE(n_components=2, perplexity=min(30, len(X) // 3), random_state=0, init="pca").fit_transform(X)
    return pca_2d, tsne_2d, evaluate(pca_2d, y, n_clusters), evaluate(tsne_2d, y, n_clusters)


def plot(pca_2d, tsne_2d, y, out="pca_vs_tsne.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, title in zip(axes, (pca_2d, tsne_2d), ("PCA", "t-SNE")):
        sc = ax.scatter(data[:, 0], data[:, 1], c=y, cmap="tab10", s=12)
        ax.set_title(title)
    fig.legend(*sc.legend_elements(), title="writer", loc="center right")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    return out


def demo():
    X, y = make_synthetic_writers()
    pca_2d, tsne_2d, pca_metrics, tsne_metrics = run_pipeline(X, y)

    print(f"{'metric':<6}{'PCA':>8}{'t-SNE':>8}")
    for k in pca_metrics:
        print(f"{k:<6}{pca_metrics[k]:>8.2f}{tsne_metrics[k]:>8.2f}")

    assert all(0 <= v <= 1 for v in tsne_metrics.values())
    assert tsne_metrics["NMI"] >= pca_metrics["NMI"] - 0.05, "t-SNE 分群品質意外比 PCA 差很多，檢查資料或參數"

    out = plot(pca_2d, tsne_2d, y)
    print(f"圖存到 {out}")


if __name__ == "__main__":
    demo()
