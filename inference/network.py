import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import LayerNorm

from transformer import Transformer
from coordconv import CoordConv2d, CoordConv1d


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class SignWithSigmoidGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input


class Painter(nn.Module):

    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3,
                 device="cpu"):
        super().__init__()
        self.local_encoder_t = nn.Sequential(
            nn.ReflectionPad2d(1),
            CoordConv2d(3, 32, 3, 1, with_r=True, use_cuda=device),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.local_encoder_c = nn.Sequential(
            nn.ReflectionPad2d(1),
            CoordConv2d(3, 32, 3, 1, with_r=True, use_cuda=device),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))

        self.local_encoder_d = nn.Sequential(
            nn.ReflectionPad2d(1),
            CoordConv2d(3, 32, 3, 1, with_r=True, use_cuda=device),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv = nn.Conv2d(128 * 3, hidden_dim, 1)

        self.sub_conv1 = nn.Conv1d(128, 256, 1, 1)
        self.sub_conv2 = nn.Conv1d(64, 8, 1, 1)

        self.DQ_transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers, batch_first=True)
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke))
        self.linear_decider = nn.Linear(hidden_dim, 1)

    def forward(self, img, canvas, cha):
        b, _, H, W = img.shape
        It = self.local_encoder_t(img)
        Ic = self.local_encoder_c(canvas)
        Isub = self.local_encoder_d(abs(cha))
        h, w = 8, 8

        feat = torch.cat([It, Ic, Isub], dim=1)
        feat_conv = self.conv(feat)
        feat_conv = feat_conv.flatten(2).permute(0, 2, 1).contiguous()

        Isub = Isub.flatten(2)
        Isub = self.sub_conv1(Isub)
        Isub = Isub.permute(0, 2, 1)
        Isub = self.sub_conv2(Isub)

        kv = feat_conv
        hidden_state = self.DQ_transformer(kv, Isub.contiguous())
        param = self.linear_param(hidden_state)
        decision = self.linear_decider(hidden_state)
        return param, decision


def build_2d_sincos_pos_emb(h, w, dim, device):
    """[1B] 2D sinusoidal 全域位置編碼，回傳 (1, h*w, dim)。與 train 端 byte-identical。"""
    assert dim % 4 == 0, 'hidden_dim 必須可被 4 整除才能做 2D sincos 位置編碼'
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device).float(),
        torch.arange(w, device=device).float(),
        indexing='ij')
    omega = torch.arange(dim // 4, device=device).float() / (dim // 4)
    omega = 1.0 / (10000 ** omega)
    y = yy.flatten()[:, None] * omega[None, :]
    x = xx.flatten()[:, None] * omega[None, :]
    pe = torch.cat([y.sin(), y.cos(), x.sin(), x.cos()], dim=1)
    return pe[None]


class PainterGlobal(nn.Module):
    """[1B] 全域 DETR 式畫家（推論端）。forward 與 train 端 byte-identical，
    確保訓練與推論用同一套表示（含全域採色）。輸出 (b,N,d_shape+7) 與 (b,N,1)。
    """

    def __init__(self, param_per_stroke, n_queries, hidden_dim, n_heads=8,
                 n_enc_layers=3, n_dec_layers=3, extra_down=2, dq_query=True,
                 coarse_to_fine=False, device="cpu"):
        super().__init__()
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim
        self.dq_query = dq_query   # True=差異驅動 query（論文 DQ 精神）；False=純可學習 DETR query

        def make_encoder():
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                CoordConv2d(3, 32, 3, 1, with_r=True, use_cuda=device),
                nn.BatchNorm2d(32), nn.ReLU(True),
                nn.ReflectionPad2d(1), nn.Conv2d(32, 64, 3, 2),
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3, 2),
                nn.BatchNorm2d(128), nn.ReLU(True))

        self.local_encoder_t = make_encoder()
        self.local_encoder_c = make_encoder()
        self.local_encoder_d = make_encoder()
        self.conv = nn.Conv2d(128 * 3, hidden_dim, 1)

        down_layers = []
        for _ in range(extra_down):
            down_layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                            nn.BatchNorm2d(hidden_dim), nn.ReLU(True)]
        self.downsample = nn.Sequential(*down_layers) if down_layers else nn.Identity()

        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers,
                                          batch_first=True)
        self.query_embed = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        # [DQ] 差異驅動 query（防塌縮版，與 train 端 byte-identical）：N 個 query 各綁固定錨點，
        # 從差異特徵圖自己錨點 grid_sample 局部差異 → 天生多樣、又差異驅動。
        if self.dq_query:
            self.diff_to_query = nn.Conv2d(128, hidden_dim, 1)
            self.query_norm = nn.LayerNorm(hidden_dim)
            g = int(round(n_queries ** 0.5))
            assert g * g == n_queries, 'dq_query=1 需 n_queries 為完全平方數（如 400=20²）'
            # [DQ 動態錨點] 候選網格邊長取 2g（N=400→40×40=1600 候選格），
            # 每次 forward 從「當前殘差」挑 top-N 候選格當錨點 → 錨點隨殘差移動。
            self.cand_g = g * 2

        # [B coarse-to-fine] 把「尺度」純量 (0=細筆、1=粗筆) 編碼後加到 query：
        # 讓同一個模型依當前 pass 的粗細改變行為（粗 pass 鋪大色塊、細 pass 補小筆），
        # 模擬人類「先大塊打底、再層層精修」。預設關 → forward 與單尺度版完全一致。
        self.coarse_to_fine = coarse_to_fine
        if coarse_to_fine:
            self.scale_embed = nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim))

        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke))
        self.linear_decider = nn.Linear(hidden_dim, 1)

    def forward(self, img, canvas, cha, scale=0.0):
        b, _, H, W = img.shape
        It = self.local_encoder_t(img)
        Ic = self.local_encoder_c(canvas)
        Isub = self.local_encoder_d(abs(cha))

        feat = torch.cat([It, Ic, Isub], dim=1)
        feat = self.conv(feat)
        feat = self.downsample(feat)
        b, c, h, w = feat.shape

        pos = build_2d_sincos_pos_emb(h, w, c, feat.device)
        memory = feat.flatten(2).permute(0, 2, 1).contiguous() + pos

        if self.coarse_to_fine:                                   # [B] 尺度純量 → (b,1,1)
            if not torch.is_tensor(scale):
                scale = torch.full((b,), float(scale), device=img.device)
            sc = scale.view(b, 1, 1)                              # 0=細筆，1=粗筆

        anchors01 = None   # [防塌縮] 每個 query 的動態錨點座標（[0,1]），dq_query 時填，供訓練端錨點定位損失
        if self.dq_query:
            # [DQ 動態錨點] 錨點依「當前殘差」每 forward 重新佈：query 自動移到還沒畫好
            # （殘差大）處；畫好處殘差≈0 → 不佈點 → 不重畫。每 pass 殘差變 → 佈點全變。
            diff = self.diff_to_query(Isub)                      # (b, hidden, h4, w4)
            with torch.no_grad():
                sal = cha.abs().mean(1, keepdim=True)            # (b,1,H,W) 殘差顯著度
                # [細節] 錨點顯著度池化：avg=平均殘差（鋪大面積，與訓練一致）；
                #   max=每格最大殘差（小而高對比的眼/鼻不被平均稀釋、更容易搶到錨點）；blend=兩者平均。
                #   預設 avg；推論可設 net.anchor_pool='max'/'blend' 做細節實驗（免重訓）。
                ap = getattr(self, 'anchor_pool', 'avg'); g = self.cand_g
                if ap == 'max':
                    sal = nn.functional.adaptive_max_pool2d(sal, (g, g))
                elif ap == 'blend':
                    sal = 0.5 * nn.functional.adaptive_avg_pool2d(sal, (g, g)) \
                          + 0.5 * nn.functional.adaptive_max_pool2d(sal, (g, g))
                else:
                    sal = nn.functional.adaptive_avg_pool2d(sal, (g, g))
                idx = torch.topk(sal.flatten(1), self.n_queries, dim=1).indices  # (b,N) 殘差最大的 N 格
                gy = (idx // self.cand_g).float()
                gx = (idx % self.cand_g).float()
                ax = (gx + 0.5) / self.cand_g * 2 - 1            # 格中心 → [-1,1]
                ay = (gy + 0.5) / self.cand_g * 2 - 1
                anchors = torch.stack([ax, ay], dim=-1)          # (b,N,2)
                # [防格紋] 錨點在所屬格內隨機抖動 ±半格 → 取樣點不對齊網格中心，
                # 加上輸出座標本來就是連續的（linear_param，非 snap 到格），雙重保證不落成格紋。
                jit = (torch.rand_like(anchors) - 0.5) * (2.0 / self.cand_g)
                anchors = (anchors + jit).clamp(-1, 1)
                anchors01 = (anchors + 1.0) * 0.5                # → [0,1]，對齊 pred 的 xc,yc（已 detach）
            grid = anchors.view(b, self.n_queries, 1, 2)
            sampled = nn.functional.grid_sample(diff, grid, align_corners=False)  # (b, hidden, N, 1)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()           # (b, N, hidden)
            queries = self.query_norm(self.query_embed[None].expand(b, -1, -1) + sampled)
        else:
            queries = self.query_embed[None].expand(b, -1, -1)

        if self.coarse_to_fine:                                   # [B] 尺度條件加到 query
            queries = queries + self.scale_embed(sc.view(b, 1)).unsqueeze(1)

        hidden_state = self.transformer(memory, queries)
        param = self.linear_param(hidden_state)
        decision = self.linear_decider(hidden_state)

        # [1B] 有界輸出（與 train 端 byte-identical）：形狀參數壓到合法範圍，w/h 恆正、座標 [0,1]。
        P = param.shape[-1]
        if P == 7:   # curved: [x0,y0,x1,y1,x2,y2, w]
            ctrl = torch.sigmoid(param[..., :6])
            width = torch.sigmoid(param[..., 6:7]) * 0.2 + 0.01
            param = torch.cat([ctrl, width], dim=-1)
        else:        # straight: [xc,yc, w(長軸), h(短軸), theta]
            xy = torch.sigmoid(param[..., :2])
            if self.coarse_to_fine:
                # [B] 尺度排程：sc=0 → 細筆做細節；sc=1 → 粗筆鋪底。
                #   長軸 w: [0.02,0.47] → [0.30,0.85]；短軸 h: [0.015,0.125] → [0.20,0.50]。
                #   下界隨 sc 抬高 → 粗 pass 強制大筆鋪底。
                w_lo = 0.02 + 0.28 * sc; w_hi = 0.47 + 0.38 * sc
                h_lo = 0.015 + 0.185 * sc; h_hi = 0.125 + 0.375 * sc
                w_len = torch.sigmoid(param[..., 2:3]) * (w_hi - w_lo) + w_lo
                h_wid = torch.sigmoid(param[..., 3:4]) * (h_hi - h_lo) + h_lo
            else:
                w_len = torch.sigmoid(param[..., 2:3]) * 0.45 + 0.02    # 長軸 [0.02,0.47]（與 c2f sc=0 一致）
                h_wid = torch.sigmoid(param[..., 3:4]) * 0.11 + 0.015   # 短軸 [0.015,0.125]
            th = torch.sigmoid(param[..., 4:5])
            param = torch.cat([xy, w_len, h_wid, th], dim=-1)

        s = self.n_queries
        grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()

        return torch.cat([param, color, color, torch.ones(b, s, 1, device=img.device)], dim=-1), decision, anchors01
