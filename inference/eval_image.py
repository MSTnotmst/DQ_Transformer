#!/usr/bin/env python
"""對「已存在的成品圖檔」直接算品質指標（不重畫推論）。

用途：當某顆 checkpoint 的最佳結果是用「舊版 code」畫出來的（例如 dyn_c2f_anc 的 0620 圖），
用現在的 code 重跑 eval 會得到不一樣（且較差）的圖，導致「報告的圖」和「eval 的數字」對不上。
本腳本直接對**已存好的圖檔**算分，保證圖與數字一致。

指標與 `eval_quality.py` **完全相同**（重用其 load_img_hwc / to_chw_01 / ssim_np / make_clipiqa、
LPIPS 餵 *2-1、PSNR=10log10(1/mse)），所以可直接跟 eval_quality 的表並列。

用法（inference/ 目錄，conda dq_transformer）：
  python eval_image.py --target ../pics/cute_ala.jpg \
    --pred output/0620_dyn__cnf_anc/cute_ala_500.jpg quality_out/baseline_patch/cute_ala.jpg \
    --labels ours_0620 baseline_3A3C --resize 512
"""
import argparse
import math
import os

import numpy as np
import torch
import lpips

import eval_quality as eq    # 重用同一套指標函式，確保與 eval_quality 的數字可比


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, help='原圖（目標）')
    ap.add_argument('--pred', nargs='+', required=True, help='一或多張已畫好的成品圖')
    ap.add_argument('--labels', nargs='*', help='對應每張 pred 的標籤（不給則用檔名）')
    ap.add_argument('--resize', type=int, default=512)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    S = (args.resize, args.resize)
    tgt = eq.load_img_hwc(args.target, S)                 # [H,W,3] in [0,1]
    tgt_t = eq.to_chw_01(tgt, device)

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    clipiqa_fn = eq.make_clipiqa(device)
    labels = args.labels if (args.labels and len(args.labels) == len(args.pred)) \
        else [os.path.basename(p) for p in args.pred]

    if clipiqa_fn:
        print('原圖 input CLIP-IQA = %.4f' % float(clipiqa_fn(tgt_t)))
    print('%-18s | L1     | PSNR  | SSIM   | LPIPS_down | CLIP-IQA_up' % 'label')
    print('-' * 74)
    for p, lab in zip(args.pred, labels):
        out = eq.load_img_hwc(p, S)
        l1 = float(np.mean(np.abs(out - tgt)))
        mse = float(np.mean((out - tgt) ** 2))
        psnr = 99.0 if mse < 1e-12 else float(10 * math.log10(1.0 / mse))
        ss = eq.ssim_np(out, tgt)
        with torch.no_grad():
            d = float(lpips_fn(eq.to_chw_01(out, device) * 2 - 1, tgt_t * 2 - 1).item())
            iqa = float(clipiqa_fn(eq.to_chw_01(out, device))) if clipiqa_fn else float('nan')
        print('%-18s | %.4f | %5.2f | %.4f | %.4f     | %.4f' % (lab, l1, psnr, ss, d, iqa))


if __name__ == '__main__':
    main()
