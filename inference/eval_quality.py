"""
eval_quality.py — 油畫「品質」評估（3A+1C 之後用）
============================================================
為什麼不只看 PSNR？
  PSNR/L1/SSIM 量的是「跟輸入逐像素有多像」。越像 = 越接近照片複製 = 越不油畫。
  一旦之後做 item 2（降 pixel 權重 + 風格損失），這些數字一定會掉——那是預期，不是退步。
  所以本腳本把指標分成「不同軸」，不要用單一數字下結論：

  [A 保真度 / 內容下限]  LPIPS(主) + PSNR/SSIM(只當 sanity floor)
                          → 不是越高越好；當「沒糊成一團」的下限，且要配筆觸數一起看。
  [B 油畫感 / 無參考]    CLIP-IQA：不需要參考圖，越高越好。獎勵「好看」而非「像原圖」。
  [C 風格距離 / 需油畫集] FID / KID（所有 outputs 的分佈 vs --oil_dir 真油畫）：越低越像油畫。
  [D 效率]               有效筆觸數（給 --count_strokes 且有 extract_strokes.py 時才算）。

每個 model 各自帶 --curved / --real_brush（與訓練 flag 對齊），可一次比多個 checkpoint。

用法（在 inference/ 目錄，conda dq_transformer）：
  python eval_quality.py \
    --input ../pics/cute_ala.jpg ../pics/NCCU.jpeg \
    --models painter.pth ../train/painter_real/latest_net_g.pth \
    --labels baseline 3A1C \
    --curved   0 1 \
    --real_brush 0 1 \
    --resize 512 --out_dir quality_out \
    --oil_dir ../pics/oil_ref          # 可選：真油畫資料夾，給 C 軸 FID/KID

依賴：lpips（必要）；piq（CLIP-IQA，可選）；torchmetrics（FID/KID，可選）。缺了會自動略過該軸。
"""
import os
import math
import glob
import argparse
import numpy as np
from PIL import Image, ImageOps

import torch
import lpips
from scipy.ndimage import gaussian_filter

import inference as inf
import inference_global as infg


# ──────────────────────────────────────────────────────────────────────────────
# 基礎工具
# ──────────────────────────────────────────────────────────────────────────────
def ssim_np(a, b, C1=0.01 ** 2, C2=0.03 ** 2):
    """灰階 SSIM；a,b: [H,W,3] in [0,1]，回傳純量 [0,1]（越高越像）。"""
    ga = a.mean(2).astype(np.float64)
    gb = b.mean(2).astype(np.float64)
    mu_a = gaussian_filter(ga, 1.5)
    mu_b = gaussian_filter(gb, 1.5)
    va = gaussian_filter(ga * ga, 1.5) - mu_a ** 2
    vb = gaussian_filter(gb * gb, 1.5) - mu_b ** 2
    vab = gaussian_filter(ga * gb, 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(np.clip(s, 0, 1).mean())


def load_img_hwc(path, size_hw=None):
    """讀圖→RGB→(可選 resize 到 size_hw=(H,W))→[0,1] 的 numpy [H,W,3]。含 EXIF 轉正。"""
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if size_hw is not None:
        img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def to_chw_01(arr_hwc, device):
    """[H,W,3] in [0,1] -> [1,3,H,W] in [0,1] tensor。"""
    return torch.from_numpy(arr_hwc).permute(2, 0, 1).unsqueeze(0).to(device)


def parse_bool_list(values, n, name):
    """把 --curved / --real_brush 的 0/1 list 補/驗到長度 n。給單一值則廣播。"""
    if values is None:
        return [False] * n
    vals = [str(v).lower() in ('1', 'true', 'yes', 'y') for v in values]
    if len(vals) == 1:
        return vals * n
    assert len(vals) == n, f'--{name} 數量要等於 --models（或只給一個廣播）'
    return vals


def detect_curved(model_path):
    """從 checkpoint 的 linear_param 末層維度判斷 curved：7=curved(1C)、5=straight、其他=None。"""
    try:
        sd = torch.load(model_path, map_location='cpu')
        w = sd.get('linear_param.4.weight')
        if w is None:
            return None
        d = int(w.shape[0])
        return True if d == 7 else False if d == 5 else None
    except Exception as e:
        print(f'[eval] 警告：讀不到 {model_path} 的維度（{e}），改用 --curved 指定值')
        return None


def detect_arch(model_path):
    """自動辨識 checkpoint 屬於哪種模型：
    回傳 (is_global, is_dq)。
      is_global：有 'query_embed' = 全域 PainterGlobal（否則原版 patch Painter）。
      is_dq    ：有 'diff_to_query.weight' = 差異驅動 query（全域才有意義）。
    """
    try:
        sd = torch.load(model_path, map_location='cpu')
        is_global = any(k == 'query_embed' or k.endswith('.query_embed') for k in sd)
        is_dq = any(k.endswith('diff_to_query.weight') for k in sd)
        return is_global, is_dq
    except Exception as e:
        print(f'[eval] 警告：讀不到 {model_path} 架構（{e}），當作 patch baseline')
        return False, False


# ──────────────────────────────────────────────────────────────────────────────
# 可選依賴：CLIP-IQA（B 軸）、FID/KID（C 軸）
# ──────────────────────────────────────────────────────────────────────────────
def make_clipiqa(device):
    """回傳一個 fn(img_chw01 [1,3,H,W] in [0,1]) -> float；缺套件時回傳 None。"""
    try:
        from piq import CLIPIQA
        metric = CLIPIQA(data_range=1.0).to(device).eval()

        @torch.no_grad()
        def _fn(x):
            return float(metric(x).item())
        print('[eval] CLIP-IQA: 使用 piq.CLIPIQA')
        return _fn
    except Exception as e_piq:
        try:
            from torchmetrics.multimodal import CLIPImageQualityAssessment
            metric = CLIPImageQualityAssessment(data_range=1.0).to(device)

            @torch.no_grad()
            def _fn(x):
                return float(metric(x).item())
            print('[eval] CLIP-IQA: 使用 torchmetrics')
            return _fn
        except Exception as e_tm:
            print('[eval] 跳過 B 軸 CLIP-IQA（pip install piq 或 torchmetrics 可啟用）'
                  f'\n        piq: {e_piq}\n        torchmetrics: {e_tm}')
            return None


def compute_fid_kid(fake_paths, oil_dir, device):
    """outputs 分佈 vs 真油畫分佈。回傳 (fid, kid_mean) 或 (None, None)。"""
    oil_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        oil_paths += glob.glob(os.path.join(oil_dir, '**', ext), recursive=True)
    if len(oil_paths) == 0:
        print(f'[eval] --oil_dir「{oil_dir}」找不到圖，跳過 C 軸')
        return None, None
    if len(oil_paths) < 50:
        print(f'[eval] 警告：真油畫只有 {len(oil_paths)} 張，FID 不可靠；以 KID 為準，並建議補到數百張')
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception as e:
        print(f'[eval] 跳過 C 軸 FID/KID（pip install torchmetrics）：{e}')
        return None, None

    def _u8(paths):
        imgs = []
        for p in paths:
            a = load_img_hwc(p, (299, 299))                       # FID/Inception 慣例尺寸
            imgs.append((a * 255).astype(np.uint8))
        return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()

    real = _u8(oil_paths).to(device)
    fake = _u8(fake_paths).to(device)
    fid = FrechetInceptionDistance(normalize=False).to(device)
    fid.update(real, real=True); fid.update(fake, real=False)
    fid_val = float(fid.compute().item())
    subset = max(2, min(50, len(oil_paths), len(fake_paths)))     # KID subset 不能超過樣本數
    kid = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)
    kid.update(real, real=True); kid.update(fake, real=False)
    kid_mean = float(kid.compute()[0].item())
    return fid_val, kid_mean


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', required=True, help='一或多張目標圖（建議用照片算保真）')
    ap.add_argument('--models', nargs='+', required=True, help='一或多個權重路徑')
    ap.add_argument('--labels', nargs='+', default=None, help='對應標籤')
    ap.add_argument('--curved', nargs='+', default=None, help='每個 model 是否 curved（0/1，1C）')
    ap.add_argument('--real_brush', nargs='+', default=None, help='每個 model 是否 real_brush（0/1，3A）')
    ap.add_argument('--brush_dir', default='brush', help='筆刷資料夾（real 紋理放 <brush_dir>/real/）')
    ap.add_argument('--resize', type=int, default=512, help='出圖解析度（正方形邊長）')
    ap.add_argument('--out_dir', default='quality_out', help='輸出資料夾')
    ap.add_argument('--oil_dir', default=None, help='可選：真油畫資料夾，給 C 軸 FID/KID')
    # 全域模型（PainterGlobal）參數：自動偵測到的 global checkpoint 會用這些（須與訓練一致）。
    ap.add_argument('--g_queries', type=int, default=400, help='[全域] N queries')
    ap.add_argument('--g_hidden', type=int, default=256, help='[全域] hidden dim')
    ap.add_argument('--g_blocks', type=int, default=3, help='[全域] enc/dec 層數')
    ap.add_argument('--g_extra_down', type=int, default=2, help='[全域] extra_down')
    ap.add_argument('--g_passes', type=int, default=12, help='[全域] 回饋精修趟數上限')
    args = ap.parse_args()

    labels = args.labels if args.labels else [f'm{i}' for i in range(len(args.models))]
    assert len(labels) == len(args.models), '--labels 數量要與 --models 相同'
    curved_list = parse_bool_list(args.curved, len(args.models), 'curved')
    real_list = parse_bool_list(args.real_brush, len(args.models), 'real_brush')

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    clipiqa_fn = make_clipiqa(device)

    # 原圖錨點：CLIP-IQA 是「無參考」分數，單一數字沒意義，要有比較對象。
    # 算原圖自己的 CLIP-IQA 當基準線：模型輸出的 IQA 高於它 = 看起來比照片更像「好畫」。
    input_iqa = float('nan')
    if clipiqa_fn:
        vals = [clipiqa_fn(to_chw_01(load_img_hwc(inp, (args.resize, args.resize)), device))
                for inp in args.input]
        input_iqa = float(np.mean(vals))
        print(f"[原圖 input] 平均 CLIP-IQA {input_iqa:.4f}（當無參考分數的基準線）")

    results = []
    for path, label, curved, real_brush in zip(args.models, labels, curved_list, real_list):
        # 以 checkpoint 實際維度為準，避免 --curved 與權重不符導致 load 崩潰
        detected = detect_curved(path)
        if detected is not None and detected != curved:
            print(f"[{label}] 注意：--curved={int(curved)} 與 checkpoint 不符，"
                  f"依權重自動改為 curved={int(detected)}")
            curved = detected

        # 自動辨識架構：全域(PainterGlobal) 走 inference_global；patch baseline 走 inference。
        is_global, is_dq = detect_arch(path)
        if is_global:
            print(f"[{label}] 偵測為全域模型（dq_query={int(is_dq)}, "
                  f"{'curved' if curved else 'straight'}）→ 走 inference_global")

        sub = os.path.join(args.out_dir, label)
        os.makedirs(sub, exist_ok=True)
        fake_paths, l1s, psnrs, ssims, lps, iqas = [], [], [], [], [], []

        for inp in args.input:
            base = os.path.basename(inp)
            out_path = os.path.join(sub, base)
            # 1) 出圖（帶該 model 的 gate；自動路由 global / patch）
            torch.manual_seed(0)   # real_brush 會隨機抽筆觸紋理，固定種子讓結果可重現
            if is_global:
                infg.paint_to_file(path, inp, out_path, res=args.resize,
                                   queries=args.g_queries, hidden=args.g_hidden,
                                   num_blocks=args.g_blocks, extra_down=args.g_extra_down,
                                   dq_query=is_dq, real_brush=real_brush, curved=curved,
                                   brush_dir=args.brush_dir, passes=args.g_passes, gpu=0)
            else:
                # curved 必須走 serial：param2img_parallel 寫死 view(-1, 8)，吃不下 curved 的 param_dim。
                inf.main(input_path=inp, model_path=path, output_dir=sub + os.sep,
                         need_animation=False, resize_h=args.resize, resize_w=args.resize,
                         serial=curved, gpu_id=0,
                         real_brush=real_brush, curved_stroke=curved, brush_dir=args.brush_dir)
            out = load_img_hwc(out_path)                          # [H,W,3]
            H, W = out.shape[:2]
            tgt = load_img_hwc(inp, (H, W))                       # 目標 resize 到輸出尺寸

            # 2) A 軸：保真度
            l1 = float(np.abs(out - tgt).mean())
            mse = float(((out - tgt) ** 2).mean())
            psnr = 99.0 if mse < 1e-12 else float(10 * math.log10(1.0 / mse))
            ss = ssim_np(out, tgt)
            with torch.no_grad():
                d = float(lpips_fn(to_chw_01(out, device) * 2 - 1,
                                   to_chw_01(tgt, device) * 2 - 1).item())
            # 3) B 軸：無參考油畫感
            iqa = clipiqa_fn(to_chw_01(out, device)) if clipiqa_fn else float('nan')

            l1s.append(l1); psnrs.append(psnr); ssims.append(ss); lps.append(d); iqas.append(iqa)
            fake_paths.append(out_path)
            print(f"[{label}] {base:>20} | L1 {l1:.4f} | PSNR {psnr:6.2f} | SSIM {ss:.4f} "
                  f"| LPIPS {d:.4f} | CLIP-IQA {iqa:.4f}")

        # 4) C 軸：整個 model 的 outputs 分佈 vs 真油畫
        fid = kid = None
        if args.oil_dir:
            fid, kid = compute_fid_kid(fake_paths, args.oil_dir, device)

        results.append({
            'label': label, 'curved': curved, 'real': real_brush,
            'l1': np.mean(l1s), 'psnr': np.mean(psnrs), 'ssim': np.mean(ssims),
            'lpips': np.mean(lps), 'iqa': np.nanmean(iqas), 'fid': fid, 'kid': kid,
        })

    # ── 摘要 ────────────────────────────────────────────────────────────────
    lines = []
    lines.append("油畫品質評估（多軸，勿用單一數字下結論）")
    lines.append("  A 保真度: LPIPS↓主、PSNR/SSIM↑只當下限（越高≠越好，要配筆數）")
    lines.append("  B 油畫感: CLIP-IQA↑（無參考，越高越好）")
    lines.append("  C 風格距: FID↓ / KID↓（vs 真油畫，給 --oil_dir 才有）")
    lines.append(f"  輸入 {len(args.input)} 張  resize={args.resize}\n")
    head = f"{'label':>12} | {'gate':>10} | LPIPS↓ | PSNR | SSIM | CLIP-IQA↑ | FID↓ | KID↓"
    lines.append(head)
    lines.append('-' * len(head))
    # 原圖錨點列：只有 CLIP-IQA 有意義（保真度欄是自己跟自己比，留白）
    lines.append(f"{'原圖 input':>12} | {'-':>10} | {'-':>6} | {'-':>5} | {'-':>5} | "
                 f"{input_iqa:8.4f} | {'-':>5} | {'-':>6}")
    for r in results:
        gate = ('curved' if r['curved'] else 'straight') + ('+real' if r['real'] else '')
        fid = f"{r['fid']:.2f}" if r['fid'] is not None else '  -'
        kid = f"{r['kid']:.4f}" if r['kid'] is not None else '   -'
        lines.append(f"{r['label']:>12} | {gate:>10} | {r['lpips']:.4f} | {r['psnr']:5.2f} | "
                     f"{r['ssim']:.4f} | {r['iqa']:8.4f} | {fid:>5} | {kid:>6}")
    summary = '\n'.join(lines)

    out_txt = os.path.join(args.out_dir, 'quality_summary.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print('\n' + summary)
    print('\n摘要 ->', out_txt)


if __name__ == '__main__':
    main()
