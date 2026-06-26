# DQ-Transformer · Automatic Oil Painting

> Differential Query Transformer for automatic oil painting — turn any photo into a
> stroke-by-stroke oil painting. TVCG 2026.
> [[Project Page](https://differential-query-painter.github.io/DQ-painter/)] · [[Paper](https://arxiv.org/abs/2603.27720)]

<img src="pics/merged_output.gif" width="760" alt="stroke-by-stroke painting">

A transformer observes the target, compares it with the current canvas, and predicts the
next set of brush strokes — producing expressive paintings with **fewer, less repetitive
strokes** than prior methods.

---

## ✨ Improvements in this fork

All gated by flags; **off by default → identical to the original model.**

| Flag | What it does |
|------|--------------|
| `--real_brush` | **Real brush textures (3A)** instead of two flat templates, **+ real stroke-parameter statistics (3C)** fitted from real strokes. One switch. |
| `--curved_stroke` | **Curved Bézier strokes (1C)** — one token = one continuous stroke, removing the “short straight segments” look. |

See `CLAUDE.md` for the full roadmap (incl. perceptual/style losses, real-painting GAN
supervision, and global de-gridding).

---

## 🚀 Quickstart (WSL2 · zsh · conda)

```zsh
conda create -n dqp python=3.9 -y && conda activate dqp
# install a CUDA build of torch that matches your driver (PyTorch 1.7+)
pip install torch torchvision pillow numpy scipy visdom dominate
nvidia-smi   # confirm the GPU is visible inside WSL
```

### Inference (original)
```zsh
cd inference
python inference.py            # edit input_path / model_path at the bottom of inference.py
```

### Train (original)
```zsh
cd train
bash my_train.sh              # checkpoints saved under checkpoints/painter
```

---

## 🖌️ Using the improvements

**1. Build a real brush library (3A)** — pick one:
```zsh
python tools/setup_brushes.py --synthesize 64          # starter set from built-ins
python tools/setup_brushes.py --src /path/to/brushes   # import a real dataset
```

**2. (Optional) Fit real stroke statistics (3C):**
```zsh
python tools/fit_strokes.py --src /path/to/single_strokes              # straight
python tools/fit_strokes.py --src /path/to/single_strokes --curved     # curved
```

**3. Train with the features:**
```zsh
cd train
# 3A + 3C
python my_train.py --name painter_real --model painter --dataset_mode null \
  --gpu_ids 0 --batch_size 128 --max_dataset_size 256 --real_brush

# 1C (curved strokes)
python my_train.py --name painter_curved --model painter --dataset_mode null \
  --gpu_ids 0 --batch_size 128 --max_dataset_size 256 --curved_stroke

# combined
python my_train.py --name painter_full --model painter --dataset_mode null \
  --gpu_ids 0 --batch_size 128 --max_dataset_size 256 --real_brush --curved_stroke
```

**4. Inference with the features** — edit the `main(...)` call at the bottom of
`inference/inference.py` and set `real_brush=True` and/or `curved_stroke=True`
(`curved_stroke` requires a checkpoint trained with `--curved_stroke`).

---

## Citation
```bibtex
@article{liu2026look,
  author  = "Liu, Lingyu and Wang, Yaxiong and Zhu, Li and Liao, Lizi and Zheng, Zhedong",
  title   = "Look, Compare and Draw: Differential Query Transformer for Automatic Oil Painting",
  journal = "TVCG",
  year    = "2026"
}
```

## Acknowledgments
Built on [Paint Transformer](https://github.com/Huage001/PaintTransformer) and
[Compositional Neural Painter](https://github.com/sjtuplayer/Compositional_Neural_Painter).
