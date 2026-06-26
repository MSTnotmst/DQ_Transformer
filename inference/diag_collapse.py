#!/usr/bin/env python
"""[診斷] 量化 PainterGlobal 是否 query collapse（400 個 query 是否吐同一筆）。

在 inference/ 目錄下跑（與 inference_global.py 同環境）：
    python diag_collapse.py --model ../train/painter_global_dyn_c2f/latest_net_g.pth \
        --input ../pics/cute_ala.jpg --res 256 --queries 400 --hidden 256 --num_blocks 3

判讀：
  * 若 param/logit 的「跨 query 標準差」≈ 0 → 確定 collapse（所有筆相同）。
  * 沿管線往回看 query_embed / sampled(diff) / queries / hidden_state 的跨-query std，
    哪一層 std 突然掉到 ~0，差異就是在那裡被抹平的。
"""
import argparse
import torch
import network
import stroke_render
import inference as base


def col_std(x):
    """x:(1,N,C) → 回傳跨 N 個 query 的逐維標準差的平均（純量）。"""
    return x[0].std(dim=0).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', default='../pics/cute_ala.jpg')
    ap.add_argument('--res', type=int, default=256)
    ap.add_argument('--queries', type=int, default=400)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--num_blocks', type=int, default=3)
    ap.add_argument('--extra_down', type=int, default=2)
    ap.add_argument('--dq_query', type=int, default=1)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.model, map_location=device)
    c2f = any('scale_embed' in k for k in state)
    net = network.PainterGlobal(5, args.queries, args.hidden, n_heads=8,
                                n_enc_layers=args.num_blocks, n_dec_layers=args.num_blocks,
                                extra_down=args.extra_down, dq_query=bool(args.dq_query),
                                coarse_to_fine=c2f, device=device).to(device)
    net.load_state_dict(state)
    net.eval()
    print('coarse_to_fine=%d | query_embed 跨-query std = %.5f'
          % (c2f, net.query_embed.std(dim=0).mean().item()))

    img = base.read_img(args.input, 'RGB', args.res, args.res).to(device)
    canvas = torch.zeros_like(img)
    cha = img - canvas

    # 把中間張量掛 hook，看差異在哪層死掉
    feats = {}
    if bool(args.dq_query):
        h1 = net.query_norm.register_forward_hook(lambda m, i, o: feats.update(queries=o.detach()))
    with torch.no_grad():
        param, decision, _ = net(img, canvas, cha, scale=1.0)   # pass1 用最粗尺度，與推論一致

    print('\n--- 跨 400 個 query 的標準差（≈0 = 全部相同 = collapse）---')
    print('xc      std = %.6f' % param[0, :, 0].std().item())
    print('yc      std = %.6f' % param[0, :, 1].std().item())
    print('w(長軸) std = %.6f' % param[0, :, 2].std().item())
    print('h(短軸) std = %.6f' % param[0, :, 3].std().item())
    print('theta   std = %.6f' % param[0, :, 4].std().item())
    print('logit   std = %.6f   (min/mean/max = %.4f / %.4f / %.4f)'
          % (decision[0, :, 0].std().item(),
             decision.min().item(), decision.mean().item(), decision.max().item()))
    if 'queries' in feats:
        print('\nqueries（送進 decoder 前，query_norm 之後）跨-query std = %.6f' % col_std(feats['queries']))
    print('\n參考前 5 筆 (xc,yc,w,h,theta)：')
    for i in range(5):
        print('  ', [round(param[0, i, j].item(), 4) for j in range(5)])


if __name__ == '__main__':
    main()
