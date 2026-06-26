#!/usr/bin/env python
"""[3A] Fetch REAL brush-stroke textures from public GitHub repos and install
them into the brush library used by --real_brush.

It shallow-clones one or more curated repos, harvests their brush PNGs
(using the alpha channel when present), normalises them to grayscale squares,
de-duplicates, and writes into BOTH:
    train/brush/real/      and      inference/brush/real/

Curated sources (all contain real brush/stroke textures):
    snp               jiupinjia/stylized-neural-painting          (oil-paint brushes) [recommended]
    brushstroke       CompVis/brushstroke-parameterized-style-transfer (strokes from real paintings)
    compositional     sjtuplayer/Compositional_Neural_Painter     (brush library)
    paint-transformer Huage001/PaintTransformer                   (2 baseline meta brushes)

Examples:
    python tools/fetch_brushes.py --list
    python tools/fetch_brushes.py --source snp
    python tools/fetch_brushes.py --source all
    python tools/fetch_brushes.py --source all --augment 4   # +rotations for variety

Requires: git on PATH, plus numpy and pillow (your conda env).
"""
import argparse
import glob
import hashlib
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIRS = [os.path.join(HERE, 'train', 'brush', 'real'),
             os.path.join(HERE, 'inference', 'brush', 'real')]

# name -> (git_url, [harvest globs relative to repo root])
SOURCES = {
    'snp': ('https://github.com/jiupinjia/stylized-neural-painting.git',
            ['**/brush*.png', '**/*brush*.png']),
    'brushstroke': ('https://github.com/CompVis/brushstroke-parameterized-style-transfer.git',
                    ['**/*brush*.png', '**/*stroke*.png']),
    'compositional': ('https://github.com/sjtuplayer/Compositional_Neural_Painter.git',
                      ['**/brush*.png', '**/*brush*.png']),
    'paint-transformer': ('https://github.com/Huage001/PaintTransformer.git',
                          ['**/brush/*.png']),
}


def to_alpha(path):
    """Return a single-channel float brush alpha in [0,1], or None if not brush-like."""
    try:
        im = Image.open(path)
    except Exception:
        return None
    im.load()
    if im.mode in ('RGBA', 'LA'):
        a = np.array(im.convert('RGBA'))[:, :, 3].astype(np.float32) / 255.0
        if a.max() - a.min() < 0.05:           # fully opaque -> alpha carries no shape
            a = np.array(im.convert('L')).astype(np.float32) / 255.0
            if a.mean() > 0.5:
                a = 1.0 - a
    else:
        rgb = np.array(im.convert('RGB')).astype(np.float32)
        sat = rgb.max(2) - rgb.min(2)
        if sat.mean() > 25:                    # colourful -> a preview icon, not a brush
            return None
        a = np.array(im.convert('L')).astype(np.float32) / 255.0
        if a.mean() > 0.5:                     # dark stroke on light bg -> invert
            a = 1.0 - a
    a = np.clip((a - a.min()) / (a.max() - a.min() + 1e-6), 0, 1)
    if a.sum() < (a.size * 0.002):             # essentially empty
        return None
    return a


def to_square(a, size):
    arr = (a * 255).astype(np.uint8)
    ys, xs = np.where(arr > 8)
    if len(xs):
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    s = max(h, w)
    canvas = np.zeros((s, s), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    return np.array(Image.fromarray(canvas).resize((size, size), Image.BILINEAR))


def clone(url, cache):
    dst = os.path.join(cache, hashlib.md5(url.encode()).hexdigest())
    if not os.path.isdir(dst):
        print('cloning', url)
        subprocess.run(['git', 'clone', '--depth', '1', url, dst], check=True)
    return dst


def harvest(source, cache, size):
    url, globs = SOURCES[source]
    repo = clone(url, cache)
    paths = []
    for g in globs:
        paths += glob.glob(os.path.join(repo, g), recursive=True)
    out, seen = [], set()
    for p in sorted(set(paths)):
        if p.lower().endswith('_prev.png'):
            continue
        a = to_alpha(p)
        if a is None:
            continue
        sq = to_square(a, size)
        key = hashlib.md5(sq.tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(sq)
    print('  %s -> %d brushes' % (source, len(out)))
    return out


def augment(brushes, factor, size):
    if factor <= 1:
        return brushes
    rng = np.random.default_rng(0)
    out = []
    for b in brushes:
        out.append(b)
        for _ in range(factor - 1):
            ang = float(rng.uniform(-20, 20))
            im = Image.fromarray(b).rotate(ang, resample=Image.BILINEAR, expand=False)
            out.append(np.array(im.resize((size, size), Image.BILINEAR)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='snp',
                    help="source name, comma-list, or 'all' (see --list)")
    ap.add_argument('--cache', default=os.path.join(tempfile.gettempdir(), 'dqp_brush_cache'))
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--augment', type=int, default=1, help='x rotations per brush for more variety')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        for k, (u, _) in SOURCES.items():
            print('%-18s %s' % (k, u))
        return

    names = list(SOURCES) if args.source == 'all' else [s.strip() for s in args.source.split(',')]
    os.makedirs(args.cache, exist_ok=True)
    brushes = []
    for n in names:
        if n not in SOURCES:
            print('unknown source:', n, '(use --list)'); continue
        try:
            brushes += harvest(n, args.cache, args.size)
        except subprocess.CalledProcessError:
            print('  clone failed for', n, '(check network / git)')

    if not brushes:
        raise SystemExit('No brushes harvested. Try --source all, or use '
                         'tools/setup_brushes.py --src <your_dir> with local images.')

    brushes = augment(brushes, args.augment, args.size)
    for d in DEST_DIRS:
        os.makedirs(d, exist_ok=True)
    for i, b in enumerate(brushes):
        for d in DEST_DIRS:
            Image.fromarray(b).save(os.path.join(d, 'brush_%04d.png' % i))
    print('Installed %d real brushes into:' % len(brushes))
    for d in DEST_DIRS:
        print('  ', d)


if __name__ == '__main__':
    main()
