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
         gif_path=None, gif_fps=2, gif_max=512):
    """[金字塔] 全域 coarse-to-fine：低解析度起步、逐層升 res，每層在殘差上補筆精修。

    每層**全域**（整張圖一次 forward、無 patch、無棋盤 → 去格保住）；逐層升解析度 → 高層看
    細節下小筆，複製原 patch 版金字塔的準確機制（眼/鼻/乾淨邊界）。

    img_full : (1,3,R,R) 目標圖（最終解析度）。回傳 (1,3,R,R)。
    levels   : 金字塔層數（建議 = 訓練的 --steps）；解析度 R/2^(levels-1)…R 逐層加倍、下界 64。
    passes   : 每層精修趟數上限（某趟畫面變化 < stop_delta 就提早進下一層）。
    keep_all / decision_thresh / soft_decision : decision 模式（同前；keep_all 診斷全畫）。
    """
    device = img_full.device
    res_list = [max(R // (2 ** (levels - 1 - i)), 64) for i in range(levels)]
    canvas = torch.zeros(1, 3, res_list[0], res_list[0], device=device)

    def gframe(c):                                   # GIF 影格：統一升到 R 再縮，避免各層尺寸不一
        if c.shape[-1] != R:
            c = F.interpolate(c, (R, R), mode='bilinear', align_corners=False)
        return _to_pil(c, gif_max)
    frames = [gframe(canvas)] if gif_path else None

    for li, res in enumerate(res_list):
        img_L = F.interpolate(img_full, (res, res), mode='area') if res < R else img_full
        if canvas.shape[-1] != res:                  # 上層結果升採樣帶到本層
            canvas = F.interpolate(canvas, (res, res), mode='bilinear', align_corners=False)
        scale = 1.0 - li / max(1, levels - 1)        # 低層粗(1.0) → 高層細(0.0)，與訓練同
        for p in range(passes):
            prev = canvas
            cha = img_L - canvas
            resid = cha.abs().mean().item()
            param, decision, _ = net(img_L, canvas, cha, scale=scale)
            N, d = param.shape[1], param.shape[2]
            logit = decision.view(1, N)
            if keep_all:
                dec = torch.ones(1, N, 1, 1, 1, device=device)
            elif soft_decision:
                dec = torch.sigmoid(decision).view(1, N, 1, 1, 1)
            else:
                dec = (decision > decision_thresh).float().view(1, N, 1, 1, 1)
            kept = float(dec.sum().item())
            for s in range(0, N, chunk):             # 分塊渲染省顯存（在本層 res 上渲染）
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
            delta = (canvas - prev).abs().mean().item()
            print('  L%d/%d res%d pass%d/%d：尺度 %.2f | 有效筆數 %.0f/%d | logit %.2f/%.2f/%.2f | 殘差 %.4f | 變化 %.4f'
                  % (li + 1, levels, res, p + 1, passes, scale, kept, N,
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
    img = base.read_img(args.input, 'RGB', R, R).to(device)   # 直接縮到 R×R 方形
    print('輸入 %s → (1,3,%d,%d)，金字塔 levels=%d、每層 passes=%d，N=%d'
          % (args.input, R, R, args.levels, args.passes, args.queries))

    gif_path = None
    if args.gif:
        stem = os.path.splitext(args.out_name)[0] if args.out_name else os.path.splitext(os.path.basename(args.input))[0]
        gif_path = os.path.join(args.output_dir, stem + '_process.gif')

    canvas = paint(net, img, R, args.levels, args.passes, meta_brushes, args.real_brush, args.curved, args.chunk,
                   keep_all=args.keep_all, decision_thresh=args.decision_thresh,
                   soft_decision=args.soft_decision, stop_delta=args.stop_delta,
                   gif_path=gif_path, gif_fps=args.gif_fps, gif_max=(args.gif_max or None))

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
