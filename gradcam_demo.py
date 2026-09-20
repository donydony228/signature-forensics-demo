"""Grad-CAM 示範：SigNet 判斷「像不像本人」時，是看圖片哪裡

一般分類模型的 Grad-CAM 是對「某個類別的分數」反傳梯度。SigNet 是驗證問題，
沒有分類分數，改用「跟本人 prototype 的餘弦相似度」當目標——熱區顯示的是
圖片裡哪些筆畫讓相似度上升，這才是驗證問題該解釋的東西。

抓 SigNet 最後一層卷積（conv5，8x12 的 feature map）的梯度和 activation，
加權加總畫出熱區，疊在原圖上存成 PNG。

    .venv/bin/python gradcam_demo.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import img_as_ubyte
from skimage.io import imread

from signet_demo import CANVAS, cedar_path, embed, signet
from sigver.preprocessing.normalize import preprocess_signature

# conv5 (8x12)：語意最完整，Grad-CAM 慣例抓的那層，但放大回 150x220 後只剩一大團模糊色塊
# conv2 (17x26)：解析度高一倍多，能看出特定幾筆連筆，代價是語意較淺（比較像邊緣/筆畫偵測，
#                不像 conv5 那樣已經整合成「這是不是這個人」的高階判斷）
TARGET_LAYERS = {'conv5': signet.conv_layers.conv5, 'conv2': signet.conv_layers.conv2}


def load_tensor(path):
    processed = preprocess_signature(img_as_ubyte(imread(path, as_gray=True)), CANVAS)
    return processed, torch.from_numpy(processed).view(1, 1, 150, 220).float().div(255)


def gradcam(tensor, prototype, target_layer):
    """回傳 (150,220) 的熱區，數值 0~1，1 代表這裡對相似度分數貢獻最大"""
    activation = {}

    def save_and_retain_grad(_module, _inp, out):
        out.retain_grad()  # 這層輸出是中間節點，預設不存梯度，要手動留住
        activation['value'] = out

    hook = target_layer.register_forward_hook(save_and_retain_grad)
    signet.zero_grad()
    emb = signet(tensor)
    hook.remove()

    score = F.cosine_similarity(emb, prototype.view(1, -1))
    score.backward()

    act = activation['value']                              # (1, 256, 8, 12)
    weights = act.grad.mean(dim=(2, 3), keepdim=True)       # 每個通道對分數的重要性：梯度做 GAP
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # 加權加總 + ReLU，只留「正貢獻」
    cam = F.interpolate(cam, size=(150, 220), mode='bilinear', align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy(), score.item()


def save_overlay(processed_img, cam, out_path):
    """疊圖存檔：灰階簽名（墨跡深、背景白）+ 紅色熱區，熱區愈亮代表模型愈看重那裡"""
    ink_dark_on_white = 255 - processed_img  # preprocess_signature 回傳的是反色圖，存檔前轉回來比較好認
    rgb = np.stack([ink_dark_on_white] * 3, axis=-1).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255  # 熱區顏色：純紅，用 cam 當透明度疊上去
    blended = rgb * (1 - cam[..., None] * 0.6) + red * (cam[..., None] * 0.6)
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(out_path)


if __name__ == '__main__':
    user = 1
    prototype = torch.from_numpy(
        embed([cedar_path('original', user, i) for i in range(1, 4)]).mean(axis=0)
    ).float()

    for kind, i, label in (('original', 20, 'genuine'), ('forgeries', 5, 'forged')):
        processed, tensor = load_tensor(cedar_path(kind, user, i))
        for layer_name, layer in TARGET_LAYERS.items():
            cam, score = gradcam(tensor, prototype, layer)
            out_path = f'gradcam_{label}_{layer_name}.png'
            save_overlay(processed, cam, out_path)
            print(f'{label:<8}{layer_name:>6}  相似度 {score:.3f} -> {out_path}')
