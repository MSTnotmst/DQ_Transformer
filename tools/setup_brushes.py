#!/usr/bin/env python
"""[3A] Build the real brush-texture library used by --real_brush.

It writes normalised grayscale brush PNGs into BOTH:
    train/brush/real/      and      inference/brush/real/

Two modes:
  1) --src <dir>   : import every image under <dir>, convert to grayscale,
                     auto-crop to content, pad to square, resize, and save.
                     Use this with real brush datasets (see README "3A datasets").
  2) --synthesize N: generate N starter brushes by augmenting the two built-in
                     templates (rotate / scale / elastic). Lets you run the
                     pipeline immediately; replace with real brushes later.

Examples:
    python tools/setup_brushes.py --synthesize 64
    python tools/setup_brushes.py --src ~/datasets/oil_brushstrokes
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIRS = [os.path.join(HERE, 'train', 'brush', 'real'),
             os.path.join(HERE, 'inference', 'brush', 'real')]
BUILTIN = [os.path.join(HERE, 'train', 'brush', 'brush_large_vertical.png'),
           os.path.join(HERE, 'train', 'brush', 'brush_large_horizontal.png')]


def to_gray_square(arr, size):
    """Auto-crop to non-zero content, pad to square, resize -> (size,size) uint8."""
    ys, xs = np.where(arr > 8)
    if len(xs) > 0:
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    s = max(h, w)
    canvas = np.zeros((s, s), dtype=arr.dtype)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    return np.array(Image.fromarray(canvas).resize((size, size), Image.BILINEAR))


def save_all(images, size, crop=True):
    for d in DEST_DIRS:
        os.makedirs(d, exist_ok=True)
    n = 0
    for i, arr in enumerate(images):
        out = to_gray_square(arr, size) if crop else np.array(
            Image.fromarray(arr).resize((size, size), Image.BILINEAR))
        for d in DEST_DIRS:
            Image.fromarray(out).save(os.path.join(d, 'brush_%04d.png' % i))
        n += 1
    print('Wrote %d brushes into:' % n)
    for d in DEST_DIRS:
        print('  ', d)


def from_src(src, size):
    paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.gbr'):
        paths += glob.glob(os.path.join(src, '**', ext), recursive=True)
    paths = sorted(paths)
    if not paths:
        raise SystemExit('No images found under %s' % src)
    imgs = []
    for p in paths:
        try:
            im = Image.open(p).convert('L')
        except Exception as e:
            print('skip %s (%s)' % (p, e))
            continue
        arr = np.array(im).astype(np.float32)
        # If the brush is dark-on-light, invert so "ink" is bright.
        if arr.mean() > 127:
            arr = 255.0 - arr
        imgs.append(arr.astype(np.uint8))
    return imgs


def synthesize(n, size):
    rng = np.random.default_rng(0)
    bases = [np.array(Image.open(p).convert('L')).astype(np.float32) for p in BUILTIN]
    out = []
    for i in range(n):
        base = bases[i % len(bases)].copy()
        im = Image.fromarray(base.astype(np.uint8))
        ang = float(rng.uniform(-25, 25))
        im = im.rotate(ang, resample=Image.BILINEAR, expand=True)
        arr = np.array(im).astype(np.float32)
        # random vertical/horizontal squash to vary aspect
        sx = float(rng.uniform(0.6, 1.0))
        sy = float(rng.uniform(0.6, 1.0))
        h, w = arr.shape
        im = Image.fromarray(arr.astype(np.uint8)).resize((max(1, int(w * sx)), max(1, int(h * sy))), Image.BILINEAR)
        arr = np.array(im).astype(np.float32)
        # mild multiplicative grain so textures are not identical
        grain = rng.uniform(0.85, 1.0, size=arr.shape).astype(np.float32)
        arr = np.clip(arr * grain, 0, 255).astype(np.uint8)
        out.append(arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=str, default=None, help='directory of real brush images to import')
    ap.add_argument('--synthesize', type=int, default=0, help='generate N starter brushes from built-ins')
    ap.add_argument('--size', type=int, default=256)
    args = ap.parse_args()

    if args.src:
        save_all(from_src(args.src, args.size), args.size)
    elif args.synthesize > 0:
        save_all(synthesize(args.synthesize, args.size), args.size)
    else:
        ap.error('use either --src <dir> or --synthesize N')


if __name__ == '__main__':
    main()
