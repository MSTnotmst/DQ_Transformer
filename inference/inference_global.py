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


def _apply_orient(img, param, strength, edge_gate=True):
    """把沿邊方向混入預測 theta（param[:,:,4]），修邊界毛躁。

    strength   : 對齊強度 0~1。
    edge_gate  : True（預設）→ 每筆混合量 = strength × 該處 coherence，**只在強邊**對齊、
                 平坦區完全不動（解輪廓毛躁的正解，不會像全域對齊那樣把平坦筆觸弄斷）；
                 False → 全域一律 strength（舊行為，易把平坦區弄糟）。
    角度用圓形混合（雙倍角向量內插）→ 避免 0/π 環繞時混錯方向。
    """
    if strength <= 0.0:
        return param
    theta, coh = _orient_field(img)                          # (1,1,H,W) ×2
    xc, yc = param[:, :, 0], param[:, :, 1]                  # (1,N)
    grid = torch.stack([xc * 2 - 1, yc * 2 - 1], dim=-1).unsqueeze(2)  # (1,N,1,2)

    def _samp(f):
        return F.grid_sample(f, grid, align_corners=False,
                             mode='bilinear', padding_mode='border').squeeze(1).squeeze(-1)  # (1,N)
    t_edge = _samp(theta)                                     # 沿邊方向 [0,1]
    b = strength * (_samp(coh).clamp(0, 1) if edge_gate else torch.ones_like(t_edge))  # (1,N) 每筆混合量
    t_model = param[:, :, 4]                                  # 模型預測方向 [0,1]
    # 圓形混合：θ∈[0,1]→角度 πθ，方向無向性（π 週期）→ 用雙倍角向量內插
    am, ae = t_model * (2 * math.pi), t_edge * (2 * math.pi)  # 2θ·π
    vx = (1 - b) * torch.cos(am) + b * torch.cos(ae)
    vy = (1 - b) * torch.sin(am) + b * torch.sin(ae)
    t_new = (0.5 * torch.atan2(vy, vx)) % math.pi / math.pi   # 回 [0,1]
    param = param.clone()
    param[:, :, 4] = t_new
    return param


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
         orient_edge_gate=True):
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
    """
    device = img_full.device
    res_list = [max(R // (2 ** (levels - 1 - i)), 64) for i in range(levels)]
    canvas = torch.zeros(1, 3, res_list[0], res_list[0], device=device)

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
            param = _apply_orient(img_L, param, orient_blend, edge_gate=orient_edge_gate)
        N, d = param.shape[1], param.shape[2]
        if force_cover or keep_all:
            dec = torch.ones(1, N, 1, 1, 1, device=device)
        elif soft_decision:
            dec = torch.sigmoid(decision).view(1, N, 1, 1, 1)
        else:
            dec = (decision > decision_thresh).float().view(1, N, 1, 1, 1)
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
                  soft_decision=False):
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
                   stop_delta=stop_delta, soft_decision=soft_decision)
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
    ap.add_argument('--underpaint_passes', type=int, default=0,
                    help='[打底層] 精修前先用最粗尺度+全覆蓋鋪 N 趟色塊底（構圖上大色塊）→ 無黑洞、精修更集中。'
                         '建議 2~4。0=關閉。')
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

    gif_path = None
    if args.gif:
        stem = os.path.splitext(args.out_name)[0] if args.out_name else os.path.splitext(os.path.basename(args.input))[0]
        gif_path = os.path.join(args.output_dir, stem + '_process.gif')

    canvas = paint(net, img, R, args.levels, args.passes, meta_brushes, args.real_brush, args.curved, args.chunk,
                   keep_all=args.keep_all, decision_thresh=args.decision_thresh,
                   soft_decision=args.soft_decision, stop_delta=args.stop_delta,
                   gif_path=gif_path, gif_fps=args.gif_fps, gif_max=(args.gif_max or None),
                   orient_blend=args.orient_blend, min_scale=args.min_scale,
                   ramp_frac=args.ramp_frac, underpaint_passes=args.underpaint_passes,
                   orient_edge_gate=not args.orient_global)

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
