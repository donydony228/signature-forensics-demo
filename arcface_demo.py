"""ArcFace 微調示範：在 SigNet 特徵上加一層用 margin loss 訓練的投影

和 signet_demo.py 的差別：signet_demo.py 直接拿 SigNet 的原始特徵算餘弦相似度
（那個空間是分類訓練「順便」產生的，沒有被直接要求同人近、異人遠）。這裡額外訓
練一層 2048 -> 512 的投影，用 ArcFace loss 明確要求：同一寫者的特徵夾角要小，
不同寫者要用一個 margin 拉開角度。

只用 40 人的真簽名訓練（不碰偽造，寫者辨識本來就只需要真簽名），在完全沒看過的
15 人上（含真簽名和熟練偽造）評估，跟原始 SigNet 特徵比較，兩者用同一批人評估
才公平。

    .venv/bin/python arcface_demo.py
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signet_demo import cedar_features, eer

TRAIN_USERS = range(40)      # 訓練 ArcFace 投影用（只用真簽名）
EVAL_USERS = range(40, 55)   # 完全沒看過的人，拿來評估（open-set，模擬新使用者）
EMBED_DIM = 512


class ArcMarginHead(nn.Module):
    """ArcFace：特徵和每類的權重向量都做 L2 正規化後算 cos(theta)，對「正確類別」
    的夾角加上一個 margin m 再丟進 softmax。逼模型必須把同類的夾角壓得比
    (異類夾角 - m) 還小，才能分類正確——這就是特徵空間被要求「同人近、異人遠」
    的來源。
    """
    def __init__(self, in_dim, embed_dim, n_classes, s=30.0, m=0.5):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.weight = nn.Parameter(torch.randn(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)

    def embed(self, x):
        return F.normalize(self.proj(x), dim=-1)

    def forward(self, x, labels):
        emb = self.embed(x)
        w = F.normalize(self.weight, dim=-1)
        cos = emb @ w.t()                                    # (batch, n_classes)
        # cos(theta + m)，用三角恆等式算，避免對 acos 微分造成梯度不穩定
        # ponytail: 沒加「theta+m 超過 180 度」的 easy-margin 修正，小資料/少 epoch 碰不到，
        # 正式上線改用 insightface 等現成實作
        sin = torch.sqrt((1.0 - cos.pow(2)).clamp(min=1e-7))
        target_cos = cos * self.cos_m - sin * self.sin_m
        onehot = F.one_hot(labels, w.shape[0]).float()
        logits = self.s * (onehot * target_cos + (1 - onehot) * cos)
        return logits


def train_head(features_by_user, epochs=300, lr=0.01):
    x = np.concatenate(features_by_user)
    y = np.concatenate([[i] * len(f) for i, f in enumerate(features_by_user)])
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()

    head = ArcMarginHead(x.shape[1], EMBED_DIM, len(features_by_user))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(head(x, y), y)
        loss.backward()
        opt.step()
    print(f'  訓練完成，最終 loss {loss.item():.4f}（{len(features_by_user)} 類, {len(x)} 筆樣本, {epochs} epochs）')
    return head.eval()


def cosine(a, b):
    return a @ b / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b))


def evaluate(embed_fn, gen, forg, n_ref=3):
    """在 EVAL_USERS 上算三個指標，跟 signet_demo.py 用一樣的定義"""
    rng = np.random.RandomState(0)
    scores_gen, scores_skilled, scores_random = [], [], []
    for u in EVAL_USERS:
        idx = rng.permutation(24)
        ref = embed_fn(gen[u, idx[:n_ref]])
        test = embed_fn(gen[u, idx[n_ref:n_ref + 10]])
        others = embed_fn(gen[[v for v in EVAL_USERS if v != u]].reshape(-1, gen.shape[-1]))
        prototype = ref.mean(axis=0)
        scores_gen.append(cosine(test, prototype))
        scores_skilled.append(cosine(embed_fn(forg[u]), prototype))
        scores_random.append(cosine(others, prototype))
    global_eer, thr = eer(np.concatenate(scores_gen), np.concatenate(scores_skilled))
    user_eer = np.mean([eer(g, f)[0] for g, f in zip(scores_gen, scores_skilled)])
    far_random = np.mean(np.concatenate(scores_random) >= thr)
    return global_eer, user_eer, far_random


if __name__ == '__main__':
    gen, forg = cedar_features()

    print(f'用 {len(list(TRAIN_USERS))} 人的真簽名訓練 ArcFace 投影層（不碰偽造）...')
    head = train_head([gen[u] for u in TRAIN_USERS])

    def arcface_embed(feat):
        with torch.no_grad():
            return head.embed(torch.from_numpy(feat).float()).numpy()

    print(f'\n在完全沒看過的 {len(list(EVAL_USERS))} 人上評估（open-set，新使用者）\n')
    print(f'{"方法":<26}{"EER 全域閾值":>12}{"EER 每人閾值":>12}{"隨機偽造誤收率":>14}')
    for name, embed_fn in (('SigNet 原始特徵', lambda f: f), ('SigNet + ArcFace 投影', arcface_embed)):
        g, u, r = evaluate(embed_fn, gen, forg)
        print(f'{name:<26}{g:>12.1%}{u:>12.1%}{r:>14.1%}')
