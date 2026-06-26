#!/usr/bin/env python
"""[1B P2/P3] 全域訓練骨架 smoke test：建最小 opt，跑 set_input + optimize_parameters
兩個 iter，確認全域 GT 生成、全域渲染、Hungarian 匹配、損失反傳整條鏈通。

預設 CPU、tiny 解析度，無 GPU 也能跑。在 train/ 目錄下：
    python smoke_global_train.py
    python smoke_global_train.py --cuda --res 256 --queries 400 --gt 64   # 量顯存/速度
"""
import argparse
from types import SimpleNamespace

import torch

from models.painter_global_model import PainterGlobalModel


def build_opt(args):
    # 只塞 PainterGlobalModel.__init__ / set_input / forward / 繼承的 optimize_parameters 會用到的欄位。
    return SimpleNamespace(
        gpu_ids=[0] if args.cuda else [],
        isTrain=True,
        checkpoints_dir='./checkpoint', name='smoke_global', preprocess='resize',
        # gates
        curved_stroke=args.curved, real_brush=False, brush_dir='brush',
        real_params='brush/real_params.npy',
        # 全域超參
        global_res=args.res, n_queries=args.queries, gt_strokes=args.gt,
        old_strokes=max(args.gt // 2, 2), global_hidden=args.hidden, extra_down=args.extra_down,
        render_chunk=args.render_chunk, num_blocks=2,
        # 訓練/網路
        init_type='normal', init_gain=0.02, lr=1e-4, batch_size=args.batch,
        lambda_pixel=8.0, lambda_gt=1.0, lambda_w=10.0, lambda_decision=1.0, lambda_recall=10.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=64)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--queries', type=int, default=16)
    ap.add_argument('--gt', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--extra_down', type=int, default=1)
    ap.add_argument('--render_chunk', type=int, default=32)
    ap.add_argument('--curved', action='store_true')
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--iters', type=int, default=2)
    args = ap.parse_args()

    opt = build_opt(args)
    model = PainterGlobalModel(opt)
    print('建立 PainterGlobalModel：R=%d N=%d M=%d hidden=%d curved=%s'
          % (opt.global_res, opt.n_queries, opt.gt_strokes, opt.global_hidden, opt.curved_stroke))

    for it in range(args.iters):
        model.set_input({'A_paths': ['smoke_%d' % it] * opt.batch_size})
        # epoch=1 → 不進 WGAN 分支（>200 才開）
        model.optimize_parameters(epoch=1)
        losses = model.get_current_losses()
        line = '  '.join('%s=%.4f' % (k, v) for k, v in losses.items()
                         if k in ('pixel', 'gt', 'w', 'decision', 'decision_sum', 'G'))
        print('iter %d | %s' % (it, line))
        for k, v in losses.items():
            assert v == v, '損失 %s 變 NaN！' % k   # NaN != NaN

    # 確認形狀
    assert model.rec.shape == (opt.batch_size, 3, opt.global_res, opt.global_res)
    assert model.pred_param.shape == (opt.batch_size, opt.n_queries, model.d_shape)
    print('rec.shape =', tuple(model.rec.shape), ' pred_param.shape =', tuple(model.pred_param.shape))
    print('\n[1B P2/P3] 全域訓練骨架 smoke 通過 ✅')


if __name__ == '__main__':
    main()
