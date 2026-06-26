#!/usr/bin/env python
"""[3C] Fit real single-stroke images into stroke parameters and save the
distribution as brush/real_params.npy, consumed by training's --real_params.

Each input image should contain ONE brush stroke (bright stroke on dark, or
pass --invert for dark stroke on light). We recover the *shape* parameters:

  straight (d_shape=5): [xc, yc, w, h, theta]                  (image moments)
  curved   (d_shape=7): [x0,y0,x1,y1,x2,y2, w]                 (Bezier GD fit)

All coordinates are normalised to [0,1]; w,h are fractions of the stroke patch.
Output is written to train/brush/real_params.npy and inference/brush/real_params.npy.

Examples:
    python tools/fit_strokes.py --src ~/datasets/oil_brushstrokes
    python tools/fit_strokes.py --src ./strokes --curved
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = [os.path.join(HERE, 'train', 'brush', 'real_params.npy'),
       os.path.join(HERE, 'inference', 'brush', 'real_params.npy')]


def load_alpha(path, invert, size=128):
    im = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    a = np.array(im).astype(np.float32) / 255.0
    if invert:
        a = 1.0 - a
    elif a.mean() > 0.5:           # auto: assume stroke is the minority/darker region
        a = 1.0 - a
    a = np.clip((a - a.min()) / (a.max() - a.min() + 1e-6), 0, 1)
    return a                       # (size,size) in [0,1]


def fit_straight(a):
    """Image-moments -> [xc, yc, w, h, theta] in [0,1]."""
    H, W = a.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    m = a.sum() + 1e-6
    cx = (xs * a).sum() / m
    cy = (ys * a).sum() / m
    dx = xs - cx
    dy = ys - cy
    cxx = (dx * dx * a).sum() / m
    cyy = (dy * dy * a).sum() / m
    cxy = (dx * dy * a).sum() / m
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    evals, evecs = np.linalg.eigh(cov)            # ascending
    major = evecs[:, 1]
    theta = np.arctan2(major[1], major[0])        # radians
    theta = (theta % np.pi) / np.pi               # -> [0,1)
    long_len = 4.0 * np.sqrt(max(evals[1], 1e-6)) / W
    short_len = 4.0 * np.sqrt(max(evals[0], 1e-6)) / H
    w = float(np.clip(long_len, 0.05, 0.95))
    h = float(np.clip(short_len, 0.03, 0.95))
    return np.array([cx / W, cy / H, w, h, theta], dtype=np.float32)


def fit_curved(a, iters=120, lr=0.05):
    import torch
    from stroke_render import render_curved
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    H, W = a.shape
    target = torch.from_numpy(a).to(dev)[None, None]            # 1,1,H,W
    s = fit_straight(a)                                         # init from moments
    cx, cy, ww, hh, th = s
    ang = th * np.pi
    dxv, dyv = np.cos(ang) * ww * 0.5, np.sin(ang) * ww * 0.5
    init = np.array([cx - dxv, cy - dyv, cx, cy, cx + dxv, cy + dyv, max(hh, 0.05)], dtype=np.float32)
    p = torch.tensor(init, device=dev, requires_grad=True)
    color = torch.tensor([1., 1., 1.], device=dev)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(iters):
        param = torch.cat([p, color])[None]                    # 1,10
        _, alpha = render_curved(param, H, W, color_gradient=False)
        loss = (alpha[:, :1] - target).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    out = p.detach().cpu().numpy()
    out[:6] = np.clip(out[:6], 0.0, 1.0)
    out[6] = float(np.clip(abs(out[6]), 0.03, 0.6))
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='directory of single-stroke images')
    ap.add_argument('--curved', action='store_true', help='fit curved (d_shape=7) instead of straight (5)')
    ap.add_argument('--invert', action='store_true', help='stroke is dark on light background')
    ap.add_argument('--limit', type=int, default=0, help='cap number of strokes (0 = all)')
    args = ap.parse_args()

    if args.curved:
        import sys
        sys.path.insert(0, os.path.join(HERE, 'inference'))    # for stroke_render

    paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
        paths += glob.glob(os.path.join(args.src, '**', ext), recursive=True)
    paths = sorted(paths)
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit('No images found under %s' % args.src)

    params = []
    for i, p in enumerate(paths):
        a = load_alpha(p, args.invert)
        if a.sum() < 1.0:
            continue
        params.append(fit_curved(a) if args.curved else fit_straight(a))
        if (i + 1) % 20 == 0:
            print('fitted %d/%d' % (i + 1, len(paths)))
    arr = np.stack(params, 0).astype(np.float32)
    for o in OUT:
        os.makedirs(os.path.dirname(o), exist_ok=True)
        np.save(o, arr)
    print('Saved %s (shape %s) to:' % ('curved' if args.curved else 'straight', arr.shape))
    for o in OUT:
        print('  ', o)


if __name__ == '__main__':
    main()
