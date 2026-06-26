"""Shared stroke rendering helpers used by BOTH train/ and inference/.

Keep this file byte-identical in train/models/stroke_render.py and
inference/stroke_render.py so the renderer is consistent across training and
inference. It deliberately uses no package-relative imports.

Features:
  * load_meta_brushes(): built-in 2 templates, or a real brush library (3A).
  * render_curved(): differentiable quadratic-Bezier "tube" renderer (1C).
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _read_gray(path, device):
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].to(device)  # 1,1,H,W


def load_meta_brushes(brush_dir, device, real_brush=False, size=256):
    """Return brush alpha templates as a (N,1,H,W) tensor in [0,1].

    real_brush=False -> built-in [vertical, horizontal] (original behaviour).
    real_brush=True  -> every PNG under ``<brush_dir>/real/`` (3A). Falls back
                        to the built-in templates if that folder is empty.
    """
    if real_brush:
        paths = sorted(glob.glob(os.path.join(brush_dir, 'real', '*.png')))
        if paths:
            out = []
            for p in paths:
                b = _read_gray(p, device)
                b = F.interpolate(b, (size, size), mode='bilinear', align_corners=False)
                out.append(b)
            print('[stroke_render] loaded %d real brushes from %s/real' % (len(out), brush_dir))
            return torch.cat(out, dim=0)
        print('[stroke_render] real_brush=True but no PNG under %s/real; using built-in templates.' % brush_dir)
    v = _read_gray(os.path.join(brush_dir, 'brush_large_vertical.png'), device)
    h = _read_gray(os.path.join(brush_dir, 'brush_large_horizontal.png'), device)
    return torch.cat([v, h], dim=0)


def render_curved(param, H, W, n_samples=40, color_gradient=True, width_scale=1.0, eps_scale=1.5):
    """Differentiable quadratic-Bezier stroke renderer (1C).

    param layout (shape part first):
        [x0, y0, x1, y1, x2, y2, w, <colors...>]
        colors = [R0,G0,B0,R2,G2,B2] if color_gradient else [R,G,B]
    Coordinates are normalised to [0,1] (x along width, y along height).

    Returns (foreground (b,3,H,W), alphas (b,3,H,W)); alpha is soft in [0,1].
    """
    b = param.shape[0]
    device = param.device
    x0, y0, x1, y1, x2, y2, w = [param[:, i] for i in range(7)]

    t = torch.linspace(0, 1, n_samples, device=device)        # (n,)
    omt = 1.0 - t
    a0 = (omt * omt)[None]                                     # (1,n)
    a1 = (2 * omt * t)[None]
    a2 = (t * t)[None]
    px = a0 * x0[:, None] + a1 * x1[:, None] + a2 * x2[:, None]  # (b,n)
    py = a0 * y0[:, None] + a1 * y1[:, None] + a2 * y2[:, None]

    ys = (torch.arange(H, device=device).float() + 0.5) / H
    xs = (torch.arange(W, device=device).float() + 0.5) / W
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')            # (H,W)
    gx = gx.reshape(1, 1, H * W)
    gy = gy.reshape(1, 1, H * W)

    dx = gx - px[:, :, None]                                   # (b,n,HW)
    dy = gy - py[:, :, None]
    d2 = dx * dx + dy * dy
    dmin, idx = d2.min(dim=1)                                  # (b,HW)
    dmin = torch.sqrt(dmin + 1e-12)

    radius = (w.abs() * 0.5 * width_scale).clamp(min=1e-3)[:, None]   # (b,1)
    eps = eps_scale / max(H, W)
    alpha = torch.sigmoid((radius - dmin) / eps).view(b, 1, H, W)     # soft matte

    tmin = (idx.float() / max(n_samples - 1, 1)).view(b, 1, H, W)
    if color_gradient:
        R0, G0, B0, R2, G2, B2 = [param[:, 7 + i] for i in range(6)]
    else:
        R0, G0, B0 = [param[:, 7 + i] for i in range(3)]
        R2, G2, B2 = R0, G0, B0

    def lerp(c0, c2):
        return c0[:, None, None, None] * (1 - tmin) + c2[:, None, None, None] * tmin

    color = torch.cat([lerp(R0, R2), lerp(G0, G2), lerp(B0, B2)], dim=1)  # (b,3,H,W)
    foreground = color * alpha
    return foreground, alpha.repeat(1, 3, 1, 1)
