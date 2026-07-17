#!/usr/bin/env python
"""[1B P4] 全域去格推論：單尺度全域 + 回饋精修，無 patch、無棋盤。

與原 ``inference.py`` 的差異：
  原版 = 金字塔逐層把畫布切成 32×32 patch、逐 patch 預測、棋盤混合 → 網格。
  本檔 = 整張圖一次進 ``PainterGlobal``，輸出全域座標筆觸，單一全域畫布合成；
         細節靠「canvas 回饋重跑 K 趟」累積，不切 patch → 構造上無網格。

渲染與訓練端 byte-identical（同一套 param2stroke / render_curved），避免 train/infer 不一致。

用法（inference/ 目錄下）：
    # 真出圖（需先有訓練好的 checkpoint）
    python inference_global.py --model ../train/checkpoint/painter_global/latest_net_g.pth \
        --input ../pics/cute_ala.jpg --res 512 --queries 400 --hidden 256 --num_blocks 3 --passes 3
    # smoke（不給 --model，用隨機權重，只驗管線會跑、出圖尺寸對）
    python inference_global.py --smoke --res 256 --queries 200
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import network
import morphology
import stroke_render
import inference as base  # 借用 read_img / save_img


def _load_letterbox(path, res, pad_mode='replicate'):
    """載圖並 letterbox-pad 成 res×res 正方形（保持原始比例，不拉伸）。

    pad_mode='replicate'（預設）：用**邊緣色延伸**補邊條，不是純黑。
        原因：純黑補邊時模型把黑邊當目標 → 在上下狂下黑大筆、且邊界粗筆跨越黑邊/內容
        把黑帶進真實畫面（GIF 觀察）。replicate 讓邊條≈相鄰內容色 → 邊界筆觸不再帶黑，
        邊條在裁切後丟棄。'black' 保留舊行為（除錯用）。

    Returns:
        img  : (1,3,res,res) tensor in [0,1]
        crop : (y1, x1, y2, x2) — 有效像素區域，用於推論後裁回原始比例存檔。
    """
    img_pil = Image.open(path).convert('RGB')
    orig_w, orig_h = img_pil.size
    scale = res / max(orig_w, orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)
    img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
    off_x, off_y = (res - new_w) // 2, (res - new_h) // 2
    core = torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float().unsqueeze(0) / 255.
    pad_l, pad_r = off_x, res - new_w - off_x
    pad_t, pad_b = off_y, res - new_h - off_y
    if pad_mode == 'black':
        img = F.pad(core, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)
    else:                                            # replicate：邊緣色延伸（消除黑邊污染）
        img = F.pad(core, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
    return img, (off_y, off_x, off_y + new_h, off_x + new_w)


def _orient_field(img):
    """結構張量求每像素「沿邊方向 theta ∈ [0,1]」與「邊強度 coherence ∈ [0,1]」。

    theta：0=水平、0.5=垂直（沿邊 = 梯度垂直方向），引導筆觸沿輪廓/毛髮走。
    coherence = (λ1−λ2)/(λ1+λ2)：強且方向明確的邊（如前景/背景輪廓）→ 接近 1；
                平坦或雜訊區 → 接近 0。用來「只在強邊」引導方向、不動平坦區。
    img: (1,3,H,W)  →  (theta (1,1,H,W), coherence (1,1,H,W))
    """
    gray = img.mean(dim=1, keepdim=True)
    device, dtype = img.device, img.dtype
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                      device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    J11 = F.avg_pool2d(gx * gx, 9, stride=1, padding=4)      # 局部平均結構張量（9×9）
    J12 = F.avg_pool2d(gx * gy, 9, stride=1, padding=4)
    J22 = F.avg_pool2d(gy * gy, 9, stride=1, padding=4)
    theta_grad = 0.5 * torch.atan2(2 * J12, J11 - J22 + 1e-8)
    theta = ((theta_grad + math.pi / 2) % math.pi) / math.pi  # 沿邊方向 [0,1]
    aniso = torch.sqrt((J11 - J22) ** 2 + 4 * J12 * J12)      # λ1−λ2
    coherence = aniso / (J11 + J22 + 1e-6)                    # ∈ [0,1]，強邊→1
    return theta, coherence


def _apply_orient(img, param, strength, edge_gate=True, theta_field=None, strength_field=None):
    """把沿邊方向混入預測 theta（param[:,:,4]），修邊界毛躁。

    strength   : 對齊強度 0~1。
    edge_gate  : True（預設）→ 每筆混合量 = strength × 該處 coherence，**只在強邊**對齊、
                 平坦區完全不動（解輪廓毛躁的正解，不會像全域對齊那樣把平坦筆觸弄斷）；
                 False → 全域一律 strength（舊行為，易把平坦區弄糟）。
    theta_field/strength_field : [語意分區] 給定時改用預算好的**逐區朝向場/強度場**（由
                 _region_orient_fields 產生，已含 orient_blend 與逐區 strength、跨界羽化）；
                 此時忽略 edge_gate 與 img 的即時結構張量。不給則走上述現行每像素路徑。
    角度用圓形混合（雙倍角向量內插）→ 避免 0/π 環繞時混錯方向。
    """
    if strength <= 0.0:
        return param
    xc, yc = param[:, :, 0], param[:, :, 1]                  # (1,N)
    grid = torch.stack([xc * 2 - 1, yc * 2 - 1], dim=-1).unsqueeze(2)  # (1,N,1,2)

    def _samp(f):
        return F.grid_sample(f, grid, align_corners=False,
                             mode='bilinear', padding_mode='border').squeeze(1).squeeze(-1)  # (1,N)
    if theta_field is not None:                              # [語意分區] 逐區場已算好，直接取樣
        t_edge = _samp(theta_field)                          # 逐區朝向 [0,1]
        b = _samp(strength_field).clamp(0, 1)                # 逐區混合量（已含 strength）
    else:
        theta, coh = _orient_field(img)                      # (1,1,H,W) ×2
        t_edge = _samp(theta)                                # 沿邊方向 [0,1]
        b = strength * (_samp(coh).clamp(0, 1) if edge_gate else torch.ones_like(t_edge))  # (1,N)
    t_model = param[:, :, 4]                                  # 模型預測方向 [0,1]
    # 圓形混合：θ∈[0,1]→角度 πθ，方向無向性（π 週期）→ 用雙倍角向量內插
    am, ae = t_model * (2 * math.pi), t_edge * (2 * math.pi)  # 2θ·π
    vx = (1 - b) * torch.cos(am) + b * torch.cos(ae)
    vy = (1 - b) * torch.sin(am) + b * torch.sin(ae)
    t_new = (0.5 * torch.atan2(vy, vx)) % math.pi / math.pi   # 回 [0,1]
    param = param.clone()
    param[:, :, 4] = t_new
    return param


def _gauss_blur(x, sigma):
    """(1,1,H,W) 可分離高斯模糊，供逐區場跨界羽化（消區界接縫）。"""
    r = max(1, int(round(3 * sigma)))
    xs = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    ker = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    ker = ker / ker.sum()
    x = F.conv2d(x, ker.view(1, 1, 1, -1), padding=(0, r))
    x = F.conv2d(x, ker.view(1, 1, -1, 1), padding=(r, 0))
    return x


def _pca(x, dim=32):
    """(N,C) → (N,dim) 主成分投影（去均值 + SVD 取前 dim 個右奇異向量）。"""
    x = x - x.mean(0, keepdim=True)
    dim = min(dim, x.shape[1])
    _, _, vh = torch.linalg.svd(x, full_matrices=False)      # vh:(k,C)
    return x @ vh[:dim].T


def _kmeans_torch(x, k, iters=25, seed=0):
    """(N,C) 純 torch Lloyd K-means（免 sklearn）。回傳 (labels(N,), centroids(k,C))。"""
    n = x.shape[0]
    k = min(k, n)
    g = torch.Generator(device='cpu').manual_seed(seed)      # 固定 seed → 分區不跳動
    cen = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    lab = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        lab = torch.cdist(x, cen).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                cen[j] = x[m].mean(0)
    return lab, cen


def _merge_clusters(lab, cen, mfrac=0.35):
    """併掉「質心距離 < mfrac × 質心間平均距離」的 cluster（呼應原文合併低差異區）+ 壓實標籤。

    用相對距離而非 cosine：cosine 對非負特徵（如 RGB+xy）恆偏高會把全部併成一區；
    相對距離門檻只併真正相近的質心，過度合併時退化為「不併」而非「併成一區」。
    """
    k = cen.shape[0]
    if k <= 1:
        return lab
    d = torch.cdist(cen, cen)                                # (k,k)
    off = d[~torch.eye(k, dtype=torch.bool, device=d.device)]
    ref = float(off.mean()) + 1e-6
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(k):
        for j in range(i + 1, k):
            if float(d[i, j]) < mfrac * ref:
                parent[find(i)] = find(j)
    uniq = sorted({find(c) for c in range(k)})
    comp = {r: i for i, r in enumerate(uniq)}
    out = lab.clone()
    for c in range(k):
        out[lab == c] = comp[find(c)]
    return out


def _mode_filter(lab, win):
    """(1,1,H,W) long → win×win 多數決濾波：去孤立點、平滑區界（呼應原文去孤立點）。"""
    L = int(lab.max().item()) + 1
    oh = F.one_hot(lab[:, 0], L).permute(0, 3, 1, 2).float()  # (1,L,H,W)
    oh = F.avg_pool2d(oh, win, stride=1, padding=win // 2)
    return oh.argmax(1, keepdim=True)


_DINO = {}


def _dino_model(device):
    """快取 DINOv2 (vits14)。首次需 torch.hub 下載權重（cglab/WSL2 有網路時）。"""
    if 'm' not in _DINO:
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _DINO['m'] = m.eval().to(device)
    return _DINO['m']


def _dino_patch_features(img, device):
    """DINOv2 patch 特徵 (Np,C) + 網格尺寸 (gh,gw)。img:(1,3,H,W)∈[0,1]。"""
    m = _dino_model(device)
    side = max(14, (img.shape[-1] // 14) * 14)               # 縮到 14 的倍數
    x = F.interpolate(img, (side, side), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        f = m.forward_features((x - mean) / std)['x_norm_patchtokens'][0]  # (Np,C)
    return f, side // 14, side // 14


def _color_features(img, grid=64, pos_w=2.0):
    """[fallback] RGB + 正規化 xy 位置特徵，供無 DINOv2 時的簡易分區。"""
    x = F.interpolate(img, (grid, grid), mode='area')[0]     # (3,g,g)
    ys, xs = torch.meshgrid(torch.linspace(0, 1, grid, device=img.device),
                            torch.linspace(0, 1, grid, device=img.device), indexing='ij')
    feat = torch.cat([x.reshape(3, -1).T, xs.reshape(-1, 1), ys.reshape(-1, 1)], dim=1)  # (g*g, 5)
    feat = (feat - feat.mean(0, keepdim=True)) / (feat.std(0, keepdim=True) + 1e-6)      # 各維可比
    feat[:, 3:] *= pos_w                                      # 位置權重（>1 更依空間相鄰成區）
    return feat, grid, grid


def _semantic_regions(img, k=6, feature='dino', seed=0):
    """[語意分區] target → region label map (1,1,H,W) long。

    feature='dino'：DINOv2 patch 特徵 → PCA → K-means → 合併相似區 → 去孤立點；
    'color' 或 DINOv2 載入失敗 → RGB+xy 簡易分區（fallback）。整張只算一次（cache 在呼叫端）。
    """
    device = img.device
    feat = gh = gw = None
    if feature == 'dino':
        try:
            feat, gh, gw = _dino_patch_features(img, device)
            feat = _pca(feat, 32)
        except Exception as ex:                              # noqa: BLE001
            print('  [分區] DINOv2 載入失敗（%s）→ 回退 color' % ex)
            feat = None
    if feat is None:
        feat, gh, gw = _color_features(img)
    lab, cen = _kmeans_torch(feat, k, seed=seed)
    lab = _merge_clusters(lab, cen).view(1, 1, gh, gw).float()
    H = img.shape[-1]
    lab = F.interpolate(lab, (H, H), mode='nearest').long()  # 網格標籤 → 全解析度
    return _mode_filter(lab, win=max(3, (H // 64) | 1))


def _region_orient_fields(img, label, strength, feather):
    """[語意分區] 由 target 與 region label 算出逐區「朝向場 theta_t、強度場 smap」(1,1,H,W)。

    - 強邊處跟本地邊方向、平坦區跟該區主方向（去平坦區方向抖動 → 修毛邊，不弄斷平坦筆觸）；
    - 逐區 strength = orient_blend × 該區平均邊強度（天空等平坦區自動低、毛髮/衣褶自動高）；
    - 雙倍角向量 + 強度場一起高斯羽化 → 消區界接縫。
    """
    theta, coh = _orient_field(img)                          # (1,1,H,W) ×2
    a = theta * (2 * math.pi)
    vx_l, vy_l = torch.cos(a), torch.sin(a)                  # 本地邊雙倍角向量
    L = int(label.max().item()) + 1
    vx_r = torch.zeros_like(vx_l)
    vy_r = torch.zeros_like(vy_l)
    smap = torch.zeros_like(coh)
    for j in range(L):
        m = (label == j).float()
        denom = m.sum() + 1e-6
        w = m * coh
        sw = w.sum() + 1e-6
        vx_r = vx_r + m * ((w * vx_l).sum() / sw)            # 區主方向（coherence 加權）
        vy_r = vy_r + m * ((w * vy_l).sum() / sw)
        smap = smap + m * ((m * coh).sum() / denom)          # 區平均邊強度 → 逐區 strength
    vx = coh * vx_l + (1 - coh) * vx_r                       # 強邊跟本地、平坦跟區主方向
    vy = coh * vy_l + (1 - coh) * vy_r
    smap = strength * smap
    if feather and feather > 0:
        vx, vy, smap = _gauss_blur(vx, feather), _gauss_blur(vy, feather), _gauss_blur(smap, feather)
    theta_t = ((0.5 * torch.atan2(vy, vx)) % math.pi) / math.pi
    return theta_t, smap.clamp(0, 1)


def _save_region_png(img, label, path):
    """把 region label 上色疊在 target 上存 PNG（診斷分區是否合理，走 chat/ 截圖流程）。"""
    L = int(label.max().item()) + 1
    g = torch.Generator().manual_seed(0)
    palette = torch.rand(L, 3, generator=g).to(label.device)     # 固定配色
    col = palette[label[0, 0].long()]                            # (H,W,3)
    over = 0.5 * img[0].permute(1, 2, 0) + 0.5 * col
    arr = (over.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    Image.fromarray(arr).save(path)
    print('  [分區] 已存分區可視化：%s（%d 區）' % (path, L))


def _stroke_resid(cha, param, curved, pool_frac=32):
    """[殘差門控] 取每筆筆畫中心的「局部平均殘差」(1,N)，供只在差異大處下筆。

    cha    : (1,3,H,W) 目標 − 畫布。
    param  : (1,N,d)。直筆中心 = (param[..,0], param[..,1])；曲筆（貝茲三控制點）
             取曲線 t=0.5 點 = 0.25·p0 + 0.5·p1 + 0.25·p2。
    殘差圖先用 ~res/pool_frac 的視窗平滑 → 量的是「這一帶」的誤差，不是單一像素，
    避免筆畫中心恰好落在已畫好的像素上就被誤殺。
    """
    err = cha.abs().mean(dim=1, keepdim=True)                 # (1,1,H,W)
    k = max(3, (err.shape[-1] // pool_frac) | 1)              # 奇數視窗
    err = F.avg_pool2d(err, k, stride=1, padding=k // 2)
    if curved:
        xc = 0.25 * param[:, :, 0] + 0.5 * param[:, :, 2] + 0.25 * param[:, :, 4]
        yc = 0.25 * param[:, :, 1] + 0.5 * param[:, :, 3] + 0.25 * param[:, :, 5]
    else:
        xc, yc = param[:, :, 0], param[:, :, 1]
    grid = torch.stack([xc * 2 - 1, yc * 2 - 1], dim=-1).unsqueeze(2)  # (1,N,1,2)
    return F.grid_sample(err, grid, align_corners=False, mode='bilinear',
                         padding_mode='border').squeeze(1).squeeze(-1)  # (1,N)


def _to_pil(canvas, max_side=None):
    """(1,3,R,R) tensor → PIL Image（uint8 RGB），可選把長邊縮到 max_side 控制 GIF 大小。"""
    arr = (canvas[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
    return im


def _composite_over(canvas, fg, alpha):
    """向量化 alpha-over（與 train 端同式）：用後綴積一次疊完一個 chunk，取代逐筆迴圈。
    canvas:(1,3,R,R)  fg,alpha:(1,cs,3,R,R)。"""
    om = 1.0 - alpha
    C = torch.flip(torch.cumprod(torch.flip(om, dims=[1]), dim=1), dims=[1])
    prod_all = C[:, 0]
    ones = torch.ones_like(C[:, :1])
    T = torch.cat([C[:, 1:], ones], dim=1)
    contribution = (fg * alpha * T).sum(dim=1)
    return canvas * prod_all + contribution


def param2stroke_straight(param, H, W, meta_brushes, real_brush):
    """[1B] 直筆全域渲染：與 train 端 painter_model.param2stroke（12 維雙色）一致。

    param: (b, 12) = [xc,yc,w,h,theta, R0,G0,B0,R2,G2,B2, A]，座標為**整張畫布** [0,1]。
    回傳 (b,3,H,W) 前景與 alpha。
    """
    b = param.shape[0]
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    R0, G0, B0, R2, G2, B2, _ = param_list[5:]
    pi = math.acos(-1.0)
    sin_theta = torch.sin(pi * theta)
    cos_theta = torch.cos(pi * theta)
    if real_brush:
        index = torch.randint(0, meta_brushes.shape[0], (b,), device=param.device)
    else:
        index = torch.full((b,), -1, device=param.device)
        index[h > w] = 0
        index[h <= w] = 1
    brush = meta_brushes[index.long()]
    alphas = torch.cat([brush, brush, brush], dim=1)
    alphas = (alphas > 0).float()
    t = torch.arange(0, brush.shape[2], device=param.device).unsqueeze(0) / brush.shape[2]
    color_map = torch.stack([R0 * (1 - t) + R2 * t, G0 * (1 - t) + G2 * t, B0 * (1 - t) + B2 * t], dim=1)
    color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
    brush = brush * color_map

    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    grid = F.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
    brush = F.grid_sample(brush, grid, align_corners=False)
    alphas = F.grid_sample(alphas, grid, align_corners=False)
    return brush, alphas


def render_strokes_global(param, H, W, meta_brushes, real_brush, curved):
    """渲染分派：curved → render_curved（color_gradient=True，與 train 同）；否則直筆。"""
    if curved:
        return stroke_render.render_curved(param, H, W, color_gradient=True)
    return param2stroke_straight(param, H, W, meta_brushes, real_brush)


@torch.no_grad()
def paint(net, img_full, R, levels, passes, meta_brushes, real_brush, curved, chunk,
         keep_all=False, decision_thresh=0.0, soft_decision=False, stop_delta=0.0015,
         gif_path=None, gif_fps=2, gif_max=512, orient_blend=0.0, blur_init=False,
         blur_sigma=16, min_scale=0.0, ramp_frac=0.5, underpaint_passes=0,
         orient_edge_gate=True, resid_gate=0.0, resid_gate_rel=False, topk=0,
         semantic_regions=False, region_k=6, region_feature='dino', region_smooth=6.0,
         region_seed=0, region_png=None):
    """[金字塔] 全域 coarse-to-fine：低解析度起步、逐層升 res，每層在殘差上補筆精修。

    每層**全域**（整張圖一次 forward、無 patch、無棋盤 → 去格保住）；逐層升解析度 → 高層看
    細節下小筆，複製原 patch 版金字塔的準確機制（眼/鼻/乾淨邊界）。

    img_full : (1,3,R,R) 目標圖（最終解析度）。回傳 (1,3,R,R)。
    levels   : 金字塔層數（建議 = 訓練的 --steps）；解析度 R/2^(levels-1)…R 逐層加倍、下界 64。
    passes   : 每層精修趟數上限（某趟畫面變化 < stop_delta 就提早進下一層）。
    keep_all / decision_thresh / soft_decision : decision 模式（同前；keep_all 診斷全畫）。
    min_scale: [筆觸下限] scale 的下限（0=原行為，可到最小筆觸界）。點狀筆觸多半是 pass 太少
               造成（少數孤立小筆看起來像點；夠多趟會累積成紋理），優先加 passes 而非拉高 min_scale。
    ramp_frac: [單解析度模式，levels==1] scale 由粗漸細所佔的 pass 比例（前 ramp_frac 趟 1.0→min_scale、
               之後停在 min_scale 繼續精修）。levels==1 才生效，複製 cute_ala_500 的「單解析度 + 數百趟
               回饋精修」機制：一路穿過粗鋪底 → 逐趟在殘差上補更小筆 → 累積出細節。levels>1 = 金字塔模式。
    underpaint_passes: [打底層] 開始精修前，先用**最粗尺度 + 全覆蓋**（keep_all）鋪 N 趟色塊底
               （像人畫畫先構圖上大色塊）。整張先被色底鋪滿 → 無黑洞、精修殘差集中在細節/邊緣。0=關閉。
    resid_gate: [殘差門控] 只在差異大的地方下筆：每筆取中心局部平均殘差，低於門檻的筆直接跳過
               （alpha 歸 0）→ 已畫好的區域不再瘋狂疊補，筆數集中在殘差最大處。0=關閉。
               打底層（force_cover）不受影響；與 keep_all / soft_decision 正交（先算 decision 再乘 gate）。
    resid_gate_rel: True → 門檻改為「resid_gate × 當前畫面平均殘差」：>1 只畫比平均差的區域，
               門檻隨收斂自動變嚴，各 pass 免手調絕對值。
    topk     : [筆數預算] 過完 decision（與 resid_gate）的筆再按「筆心局部殘差」排序，每趟只畫
               前 topk 名（0=關閉，全畫）。收斂後期高殘差點變少 → 實際下筆數自動遞減、
               已畫好區域擠不進前幾名 → 「不是過門檻就畫，只畫最值得畫的」。打底層不受影響。
               soft_decision 模式下排序鍵為 殘差×不透明度。
    """
    device = img_full.device
    res_list = [max(R // (2 ** (levels - 1 - i)), 64) for i in range(levels)]
    canvas = torch.zeros(1, 3, res_list[0], res_list[0], device=device)

    # [語意分區] 整張只算一次 region label；逐 res 的朝向/強度場 lazy 快取（跨 pass 不變）。
    region_label = None
    _orient_cache = {}
    if semantic_regions and orient_blend > 0.0:
        region_label = _semantic_regions(img_full, k=region_k, feature=region_feature, seed=region_seed)
        if region_png:
            _save_region_png(img_full, region_label, region_png)

    def _orient_fields_for(img_L):
        r = img_L.shape[-1]
        if r not in _orient_cache:
            lab = F.interpolate(region_label.float(), (r, r), mode='nearest').long()
            _orient_cache[r] = _region_orient_fields(img_L, lab, orient_blend, region_smooth)
        return _orient_cache[r]

    def gframe(c):                                   # GIF 影格：統一升到 R 再縮，避免各層尺寸不一
        if c.shape[-1] != R:
            c = F.interpolate(c, (R, R), mode='bilinear', align_corners=False)
        return _to_pil(c, gif_max)
    frames = [gframe(canvas)] if gif_path else None

    def _render_pass(canvas, img_L, res, scale, force_cover):
        """一趟：net 預測 N 筆 → 依 decision 合成到 canvas。回傳 (canvas, logit, kept)。
        force_cover=True → 無視 decision 全畫（打底層用，確保整張鋪滿）。"""
        cha = img_L - canvas
        param, decision, _ = net(img_L, canvas, cha, scale=scale)
        if orient_blend > 0.0:
            if region_label is not None:                     # [語意分區] 用逐區朝向場/強度場
                tf, sf = _orient_fields_for(img_L)
                param = _apply_orient(img_L, param, orient_blend, theta_field=tf, strength_field=sf)
            else:
                param = _apply_orient(img_L, param, orient_blend, edge_gate=orient_edge_gate)
        N, d = param.shape[1], param.shape[2]
        if force_cover or keep_all:
            dec = torch.ones(1, N, 1, 1, 1, device=device)
        elif soft_decision:
            dec = torch.sigmoid(decision).view(1, N, 1, 1, 1)
        else:
            dec = (decision > decision_thresh).float().view(1, N, 1, 1, 1)
        if resid_gate > 0.0 and not force_cover:             # [殘差門控] 誤差小的地方不下筆
            e = _stroke_resid(cha, param, curved)            # (1,N) 每筆中心局部殘差
            t = resid_gate * float(cha.abs().mean().item()) if resid_gate_rel else resid_gate
            dec = dec * (e >= t).float().view(1, N, 1, 1, 1)
        if topk > 0 and not force_cover:                     # [筆數預算] 存活筆按殘差只取前 k 名
            e = _stroke_resid(cha, param, curved)            # (1,N)
            score = e.view(-1) * dec.view(-1)                # 被 decision/gate 關掉的筆不參賽
            if int((score > 0).sum().item()) > topk:
                mask = torch.zeros(N, device=device)
                mask[score.topk(topk).indices] = 1.0
                dec = dec * mask.view(1, N, 1, 1, 1)
        kept = float(dec.sum().item())
        for s in range(0, N, chunk):                 # 分塊渲染省顯存
            e = min(s + chunk, N)
            sub = param[:, s:e].reshape(-1, d).contiguous()
            fg, al = render_strokes_global(sub, res, res, meta_brushes, real_brush, curved)
            fg = F.max_pool2d(fg, 3, stride=1, padding=1)
            al = -F.max_pool2d(-al, 3, stride=1, padding=1)
            cs = e - s
            fg = fg.view(1, cs, 3, res, res)
            al = al.view(1, cs, 3, res, res)
            alpha = al * dec[:, s:e]
            canvas = _composite_over(canvas, fg, alpha)
        return canvas, decision.view(1, N), kept

    single_scale = (levels == 1)                     # [參考機制] 單解析度 + scale 隨 pass ramp
    ramp_end = max(1, int(passes * ramp_frac))       #   前 ramp_frac 趟粗→細，之後停在 min_scale

    # [打底層] 構圖上大色塊：最粗尺度 + 全覆蓋，鋪 underpaint_passes 趟 → 整張有色底、無黑洞。
    if underpaint_passes > 0:
        res0 = res_list[0]
        img0 = F.interpolate(img_full, (res0, res0), mode='area') if res0 < R else img_full
        for up in range(underpaint_passes):
            canvas, logit, kept = _render_pass(canvas, img0, res0, scale=1.0, force_cover=True)
            resid = (img0 - canvas).abs().mean().item()
            print('  [打底 %d/%d] res%d 尺度 1.00 | 全覆蓋 %.0f 筆 | 殘差 %.4f'
                  % (up + 1, underpaint_passes, res0, kept, resid))
            if frames is not None:
                frames.append(gframe(canvas))

    for li, res in enumerate(res_list):
        img_L = F.interpolate(img_full, (res, res), mode='area') if res < R else img_full
        if canvas.shape[-1] != res:                  # 上層結果升採樣帶到本層
            canvas = F.interpolate(canvas, (res, res), mode='bilinear', align_corners=False)
        raw_level = 1.0 - li / max(1, levels - 1)    # 金字塔：ramp 綁層（粗 1.0 → 細 0.0）
        for p in range(passes):
            if single_scale:                         # 單解析度：scale 隨 pass 由粗漸細（複製 cute_ala_500 機制）
                raw = max(0.0, 1.0 - p / ramp_end)
            else:
                raw = raw_level
            scale = min_scale + (1.0 - min_scale) * raw  # [筆觸下限] 收在 min_scale 不歸 0
            prev = canvas
            resid = (img_L - canvas).abs().mean().item()
            canvas, logit, kept = _render_pass(canvas, img_L, res, scale, force_cover=False)
            delta = (canvas - prev).abs().mean().item()
            print('  L%d/%d res%d pass%d/%d：尺度 %.2f | 有效筆數 %.0f/%d | logit %.2f/%.2f/%.2f | 殘差 %.4f | 變化 %.4f'
                  % (li + 1, levels, res, p + 1, passes, scale, kept, logit.shape[1],
                     logit.min(), logit.mean(), logit.max(), resid, delta))
            if frames is not None:
                frames.append(gframe(canvas))
            if not keep_all and stop_delta > 0 and delta < stop_delta:
                print('    → 本層變化 < %.4f，進下一層。' % stop_delta)
                break

    if canvas.shape[-1] != R:                        # 確保輸出在最終解析度
        canvas = F.interpolate(canvas, (R, R), mode='bilinear', align_corners=False)
    if gif_path and frames:
        dur = int(round(1000.0 / max(1e-6, gif_fps)))
        durations = [dur] * len(frames); durations[-1] = dur * 5      # 末格(成品)停久一點
        os.makedirs(os.path.dirname(gif_path) or '.', exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True)
        print('已輸出 GIF：%s（%d 格 @ %.1f fps）' % (gif_path, len(frames), gif_fps))
    return canvas


def paint_to_file(model_path, input_path, out_path, res=512, queries=400, hidden=256,
                  num_blocks=3, extra_down=2, dq_query=True, real_brush=False, curved=False,
                  brush_dir='brush', passes=4, levels=4, min_kept=8, stop_delta=0.0015, chunk=32, gpu=0,
                  soft_decision=False, resid_gate=0.0, resid_gate_rel=False):
    """程式化呼叫：載入 PainterGlobal、金字塔逐層精修出圖、存到 out_path。

    供 eval_quality.py 等共用，等同 CLI 但可直接傳參。回傳 out_path。
    （min_kept 保留簽章相容，新金字塔流程不使用。）
    """
    device = torch.device('cuda:%d' % gpu if torch.cuda.is_available() else 'cpu')
    d_shape = 7 if curved else 5
    state = torch.load(model_path, map_location=device)
    coarse_to_fine = any('scale_embed' in k for k in state)   # [B] 自動偵測 checkpoint 是否含尺度排程
    net = network.PainterGlobal(d_shape, queries, hidden, n_heads=8,
                                n_enc_layers=num_blocks, n_dec_layers=num_blocks,
                                extra_down=extra_down, dq_query=bool(dq_query),
                                coarse_to_fine=coarse_to_fine, device=device).to(device)
    net.load_state_dict(state)
    net.eval()
    meta_brushes = stroke_render.load_meta_brushes(brush_dir, device, real_brush=real_brush)
    img = base.read_img(input_path, 'RGB', res, res).to(device)
    canvas = paint(net, img, res, levels, passes, meta_brushes, real_brush, curved, chunk,
                   stop_delta=stop_delta, soft_decision=soft_decision,
                   resid_gate=resid_gate, resid_gate_rel=resid_gate_rel)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    base.save_img(canvas[0], out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='PainterGlobal checkpoint（.pth）；不給則 --smoke 隨機權重')
    ap.add_argument('--smoke', action='store_true', help='隨機權重跑通管線（不需 checkpoint）')
    ap.add_argument('--input', default='../pics/cute_ala.jpg')
    ap.add_argument('--output_dir', default='output_global/')
    ap.add_argument('--out_name', default=None,
                    help='自訂輸出檔名（不含路徑；不給則用輸入檔名，會覆蓋舊圖）。'
                         '用它保留每次試推論結果，如 --out_name ala_prog_e50_th-1')
    ap.add_argument('--res', type=int, default=512, help='最終工作解析度 R（可比訓練大）')
    ap.add_argument('--levels', type=int, default=4, help='[金字塔] 層數（建議 = 訓練 --steps）；res 從 R/2^(levels-1) 逐層升到 R')
    ap.add_argument('--passes', type=int, default=4, help='[金字塔] 每層精修趟數上限（變化 < stop_delta 提早進下一層）')
    ap.add_argument('--stop_delta', type=float, default=0.0015, help='某層某趟畫面變化 < 此值就進下一層；0=每層跑滿 passes')
    # 下列需與訓練設定一致
    ap.add_argument('--queries', type=int, default=400)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--num_blocks', type=int, default=3, help='enc/dec 層數，須與訓練 --num_blocks 相同')
    ap.add_argument('--extra_down', type=int, default=2)
    ap.add_argument('--dq_query', type=int, default=1, help='[DQ] 須與訓練一致：1=差異驅動 query, 0=可學習')
    ap.add_argument('--coarse_to_fine', type=int, default=0,
                    help='[B] 1=尺度排程推論（早 pass 粗筆鋪底→晚 pass 細筆）；給 --model 時自動偵測，主要供 --smoke 用')
    ap.add_argument('--real_brush', action='store_true')
    ap.add_argument('--curved', action='store_true')
    ap.add_argument('--brush_dir', default='brush')
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--keep_all', action='store_true', help='診斷：無視 decision 把 N 筆全畫')
    ap.add_argument('--decision_thresh', type=float, default=0.0, help='decision logit 門檻；調負數可多畫')
    ap.add_argument('--soft_decision', action='store_true', help='[路2 模型必開] 軟決策：sigmoid 當不透明度，與訓練一致')
    ap.add_argument('--anchor_pool', default='avg', choices=['avg', 'max', 'blend'],
                    help='[細節實驗] 錨點顯著度池化：avg=平均(與訓練一致)；max=每格最大(小高對比特徵如眼/鼻更易搶到錨點)；blend=兩者平均。免重訓')
    ap.add_argument('--gif', action='store_true', help='把推論過程（每趟一格）輸出成動畫 GIF，存到 output_dir/<名>_process.gif')
    ap.add_argument('--gif_fps', type=float, default=2.0, help='GIF 播放速率（格/秒）')
    ap.add_argument('--gif_max', type=int, default=512, help='GIF 影格長邊上限（縮圖控制檔案大小；0=不縮）')
    ap.add_argument('--orient_blend', type=float, default=0.0,
                    help='[方向引導/修邊界毛躁] 沿邊方向對齊強度（0=關閉，0.4~0.7 推薦）。'
                         '預設「邊強度加權」：只在強邊（前景/背景輪廓）讓筆觸沿輪廓走→修毛邊，'
                         '平坦區不動。不需重訓。')
    ap.add_argument('--orient_global', action='store_true',
                    help='[除錯] 關掉邊強度加權，改全域一律對齊（易把平坦區弄糟，一般別開）。')
    ap.add_argument('--no_letterbox', action='store_true',
                    help='不保持原始比例（直接拉成正方形，舊行為）。預設保持比例並邊緣延伸補邊。')
    ap.add_argument('--pad_mode', default='replicate', choices=['replicate', 'black'],
                    help='letterbox 補邊方式：replicate=邊緣色延伸（預設，消除黑邊被畫成黑筆污染內容）；'
                         'black=純黑（舊行為，會有黑邊污染問題）。')
    ap.add_argument('--min_scale', type=float, default=0.0,
                    help='[筆觸下限] scale 下限（0=可到最小筆觸界）。點狀筆觸優先加 passes 解決，非拉高此值。')
    ap.add_argument('--ramp_frac', type=float, default=0.5,
                    help='[單解析度模式 --levels 1] scale 由粗漸細所佔 pass 比例，複製 cute_ala_500 機制。'
                         '搭配 --levels 1 --passes 200~500 使用。')
    ap.add_argument('--resid_gate', type=float, default=0.0,
                    help='[殘差門控/防瘋狂疊補] 只在差異大的地方下筆：每筆取中心局部平均殘差，'
                         '低於門檻直接跳過。0=關閉。絕對門檻建議 0.03~0.08；'
                         '搭 --resid_gate_rel 時為「平均殘差的倍數」（建議 1.0~2.0），隨收斂自動變嚴。'
                         '打底層（--underpaint_passes）不受影響。免重訓。')
    ap.add_argument('--resid_gate_rel', action='store_true',
                    help='[殘差門控] 門檻改為相對值：resid_gate × 當前畫面平均殘差。'
                         '>1 = 只畫比平均更差的區域，各 pass 免手調絕對門檻。')
    ap.add_argument('--topk', type=int, default=0,
                    help='[筆數預算/免重訓] 過完 decision（與 resid_gate）的筆按筆心局部殘差排序，'
                         '每趟只畫前 k 名——不是過門檻就畫，只畫最值得畫的。收斂後期下筆數自動遞減。'
                         '建議 80~150（passes 可能要加，同筆數分更多趟下）。0=關閉。打底層不受影響。')
    ap.add_argument('--underpaint_passes', type=int, default=0,
                    help='[打底層] 精修前先用最粗尺度+全覆蓋鋪 N 趟色塊底（構圖上大色塊）→ 無黑洞、精修更集中。'
                         '建議 2~4。0=關閉。')
    ap.add_argument('--semantic_regions', action='store_true',
                    help='[語意分區/借鏡 StyleGallery，免重訓] 把 --orient_blend 從全域升級成逐語意區：'
                         '先把 target 分成語意區，各區用自己的主方向與 strength 對齊筆觸（天空自動弱、'
                         '毛髮/衣褶自動強）。需搭 --orient_blend>0。0=關（走現行全域每像素路徑）。')
    ap.add_argument('--region_k', type=int, default=6, help='[語意分區] K-means 分區數上限（建議 4~6）。')
    ap.add_argument('--region_feature', default='dino', choices=['dino', 'color'],
                    help='[語意分區] 分區特徵：dino=DINOv2 語意（首次需下載權重）；color=RGB+xy（零依賴 fallback）。')
    ap.add_argument('--region_smooth', type=float, default=6.0,
                    help='[語意分區] 區界羽化高斯 sigma（消區界接縫；0=不羽化，硬邊可能出接縫）。')
    ap.add_argument('--region_seed', type=int, default=0, help='[語意分區] K-means 初始化 seed（固定→分區不跳動）。')
    args = ap.parse_args()

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    d_shape = 7 if args.curved else 5

    coarse_to_fine = bool(args.coarse_to_fine)
    state = None
    if args.model:
        state = torch.load(args.model, map_location=device)
        coarse_to_fine = any('scale_embed' in k for k in state)   # [B] 自動偵測，覆蓋 CLI
    elif not args.smoke:
        raise SystemExit('未給 --model；若只想驗管線請加 --smoke')

    net = network.PainterGlobal(d_shape, args.queries, args.hidden, n_heads=8,
                                n_enc_layers=args.num_blocks, n_dec_layers=args.num_blocks,
                                extra_down=args.extra_down, dq_query=bool(args.dq_query),
                                coarse_to_fine=coarse_to_fine, device=device).to(device)
    if state is not None:
        net.load_state_dict(state)
        print('載入 checkpoint：', args.model, '| coarse_to_fine=%d' % coarse_to_fine)
    else:
        print('[smoke] 使用隨機權重（出圖無意義，只驗管線）| coarse_to_fine=%d' % coarse_to_fine)
    net.eval()
    net.anchor_pool = args.anchor_pool      # [細節實驗] 切錨點池化（avg/max/blend），不影響權重

    meta_brushes = stroke_render.load_meta_brushes(args.brush_dir, device, real_brush=args.real_brush)

    R = args.res
    if args.smoke or args.no_letterbox:
        img = base.read_img(args.input, 'RGB', R, R).to(device)
        crop_box = None
    else:
        img, crop_box = _load_letterbox(args.input, R, pad_mode=args.pad_mode)
        img = img.to(device)
    print('輸入 %s → (1,3,%d,%d)，金字塔 levels=%d、每層 passes=%d，N=%d%s'
          % (args.input, R, R, args.levels, args.passes, args.queries,
             '' if (args.smoke or args.no_letterbox) else '（letterbox 保持比例）'))

    stem = os.path.splitext(args.out_name)[0] if args.out_name else os.path.splitext(os.path.basename(args.input))[0]
    gif_path = os.path.join(args.output_dir, stem + '_process.gif') if args.gif else None
    region_png = os.path.join(args.output_dir, stem + '_regions.png') if args.semantic_regions else None

    canvas = paint(net, img, R, args.levels, args.passes, meta_brushes, args.real_brush, args.curved, args.chunk,
                   keep_all=args.keep_all, decision_thresh=args.decision_thresh,
                   soft_decision=args.soft_decision, stop_delta=args.stop_delta,
                   gif_path=gif_path, gif_fps=args.gif_fps, gif_max=(args.gif_max or None),
                   orient_blend=args.orient_blend, min_scale=args.min_scale,
                   ramp_frac=args.ramp_frac, underpaint_passes=args.underpaint_passes,
                   orient_edge_gate=not args.orient_global,
                   resid_gate=args.resid_gate, resid_gate_rel=args.resid_gate_rel,
                   topk=args.topk,
                   semantic_regions=args.semantic_regions, region_k=args.region_k,
                   region_feature=args.region_feature, region_smooth=args.region_smooth,
                   region_seed=args.region_seed, region_png=region_png)

    # letterbox 裁回原始比例（去掉黑邊）
    if crop_box is not None:
        y1, x1, y2, x2 = crop_box
        canvas = canvas[:, :, y1:y2, x1:x2]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.out_name:
        name = args.out_name
        if not os.path.splitext(name)[1]:
            name += '.jpg'
    else:
        name = os.path.basename(args.input)
    out_path = os.path.join(args.output_dir, name)
    base.save_img(canvas[0], out_path)
    print('已輸出：', out_path)


if __name__ == '__main__':
    main()
