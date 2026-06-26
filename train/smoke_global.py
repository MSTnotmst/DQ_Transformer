#!/usr/bin/env python
"""[1B P1] PainterGlobal 前向 smoke test：餵隨機張量，檢查輸出 shape 正確、能 backward。

只驗「全域模型前向 + 形狀」，不涉訓練/資料。預設 CPU、小解析度，無 GPU 也能跑。

用法（在 train/ 目錄下）：
    python smoke_global.py
    python smoke_global.py --res 512 --batch 2 --queries 400 --cuda   # 量顯存用
"""
import argparse
import torch

from models.networks import PainterGlobal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=256, help='工作解析度 R（正方）')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--queries', type=int, default=256, help='N：全域 stroke query 數（dq_query 需平方數）')
    ap.add_argument('--d_shape', type=int, default=5, help='straight=5, curved=7')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--extra_down', type=int, default=2, help='額外 stride2 次數（總下採樣 = 4*2^extra_down）')
    ap.add_argument('--dq_query', type=int, default=1, help='1=差異驅動 query（需 queries 為平方數），0=純可學習')
    ap.add_argument('--coarse_to_fine', type=int, default=0, help='1=啟用尺度排程（以 scale=1 粗筆路徑驗前向）')
    ap.add_argument('--cuda', action='store_true')
    args = ap.parse_args()

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    print('device =', device)

    net = PainterGlobal(param_per_stroke=args.d_shape, n_queries=args.queries,
                        hidden_dim=args.hidden, extra_down=args.extra_down,
                        dq_query=bool(args.dq_query), coarse_to_fine=bool(args.coarse_to_fine),
                        device=device).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print('PainterGlobal 參數量 = %.2fM' % (n_params / 1e6))

    b, R = args.batch, args.res
    img = torch.rand(b, 3, R, R, device=device)
    canvas = torch.rand(b, 3, R, R, device=device)
    cha = img - canvas

    # 預期 token 數 = (R/4/2^extra_down)^2
    L = (R // 4 // (2 ** args.extra_down)) ** 2
    print('預期全域 memory token 數 L =', L)

    scale = 1.0 if args.coarse_to_fine else 0.0     # 粗筆路徑（驗 scale_embed + 尺度排程上下界）
    param, decision, anchors01 = net(img, canvas, cha, scale=scale)
    d_expected = args.d_shape + 7   # 形狀 + 頭尾色(3+3) + alpha(1)
    print('param.shape    =', tuple(param.shape), '  (預期 (%d, %d, %d))' % (b, args.queries, d_expected))
    print('decision.shape =', tuple(decision.shape), '  (預期 (%d, %d, 1))' % (b, args.queries))
    print('anchors01      =', None if anchors01 is None else tuple(anchors01.shape),
          '  (dq_query=1 預期 (%d, %d, 2)，=0 預期 None)' % (b, args.queries))

    assert param.shape == (b, args.queries, d_expected), 'param shape 不符！'
    assert decision.shape == (b, args.queries, 1), 'decision shape 不符！'
    if args.dq_query:
        assert anchors01 is not None and anchors01.shape == (b, args.queries, 2), 'anchors01 shape 不符！'
        assert float(anchors01.min()) >= 0.0 and float(anchors01.max()) <= 1.0, 'anchors01 應在 [0,1]！'
    else:
        assert anchors01 is None, 'dq_query=0 時 anchors01 應為 None！'

    # 確認可 backward（之後訓練要靠它）。
    loss = param.mean() + decision.mean()
    loss.backward()
    assert net.query_embed.grad is not None, 'query_embed 沒有梯度！'
    print('backward OK，query_embed 有梯度。')
    print('\n[1B P1] smoke test 全部通過 ✅')


if __name__ == '__main__':
    main()
