import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import LayerNorm
from .coordconv import CoordConv2d, CoordConv1d
from .transformer import Transformer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


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

        s = hidden_state.shape[1]
        grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()

        return torch.cat([param, color, color, torch.ones(b, s, 1, device=img.device)], dim=-1), decision


def build_2d_sincos_pos_emb(h, w, dim, device):
    """[1B] 建 2D sinusoidal 全域位置編碼，回傳 (1, h*w, dim)。

    把整張 feature map 的每個空間位置編碼成全域座標，讓 token 序列攤平後
    仍保有「它在整張圖哪裡」的資訊——這是去 patch 後 transformer 能跨「原本格界」
    對齊空間關係的關鍵。dim 需可被 4 整除（y/x 各佔 sin+cos 一半）。
    """
    assert dim % 4 == 0, 'hidden_dim 必須可被 4 整除才能做 2D sincos 位置編碼'
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device).float(),
        torch.arange(w, device=device).float(),
        indexing='ij')                                  # (h,w),(h,w)
    omega = torch.arange(dim // 4, device=device).float() / (dim // 4)
    omega = 1.0 / (10000 ** omega)                      # (dim/4,)
    y = yy.flatten()[:, None] * omega[None, :]          # (h*w, dim/4)
    x = xx.flatten()[:, None] * omega[None, :]          # (h*w, dim/4)
    pe = torch.cat([y.sin(), y.cos(), x.sin(), x.cos()], dim=1)  # (h*w, dim)
    return pe[None]                                     # (1, h*w, dim)


class PainterGlobal(nn.Module):
    """[1B] 全域 DETR 式畫家：整張圖編碼成 token 序列 + N 個可學習 stroke query
    跨全圖注意力，輸出全域座標筆觸——無 patch、無棋盤，從根本去除網格。

    與原版 ``Painter``（每 patch 獨立、8 query 只看自己 patch）並存，由
    ``--global_painter`` 切換；原版完全不受影響。

    輸入（整張圖，非 patch）：
        img, canvas, cha : (b, 3, R, R)
    輸出：
        param : (b, N, d)  其中 d = param_per_stroke + 7（形狀 + 頭尾色 + alpha）
        decision : (b, N, 1)
    """

    def __init__(self, param_per_stroke, n_queries, hidden_dim, n_heads=8,
                 n_enc_layers=3, n_dec_layers=3, extra_down=2, dq_query=True,
                 coarse_to_fine=False, device="cpu"):
        super().__init__()
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim
        self.dq_query = dq_query   # True=差異驅動 query（論文 DQ 精神）；False=純可學習 DETR query

        # 3 個編碼器結構與原版一致（CoordConv + 兩次 stride2 → 空間 /4）。
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

        # 額外 extra_down 次 stride2，把 token 數從 (R/4)² 壓到 (R/4/2^extra_down)²，
        # 控制全 attention 的成本（extra_down=2 → 總 /16，R=512 時 32×32=1024 token）。
        down_layers = []
        for _ in range(extra_down):
            down_layers += [nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
                            nn.BatchNorm2d(hidden_dim), nn.ReLU(True)]
        self.downsample = nn.Sequential(*down_layers) if down_layers else nn.Identity()

        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers,
                                          batch_first=True)
        # N 個可學習 stroke query（dq_query=True 時當「seed」，再由差異特徵調制成差異驅動 query）。
        self.query_embed = nn.Parameter(torch.randn(n_queries, hidden_dim) * 0.02)

        # [DQ 動態錨點] 差異驅動 query（防塌縮 + 隨殘差移動版）：錨點不固定，
        # 每次 forward 從「當前殘差」挑 top-N 候選格當錨點，再 grid_sample 該處局部差異 → query。
        # query 自動移到還沒畫好（殘差大）處；畫好處不佈點 → 不重畫。每 pass 殘差變 → 佈點全變。
        # 候選網格邊長 2g（N=400→40×40=1600 候選格），N 個彼此相異的格 → query 多樣、不塌縮。
        # （固定 20×20 錨點會把筆觸釘在網格、多 pass 原地重疊；先前「seed 跨注意全圖差異」則會塌縮。）
        if self.dq_query:
            self.diff_to_query = nn.Conv2d(128, hidden_dim, 1)
            self.query_norm = nn.LayerNorm(hidden_dim)
            g = int(round(n_queries ** 0.5))
            assert g * g == n_queries, 'dq_query=1 需 n_queries 為完全平方數（如 400=20²）'
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
        It = self.local_encoder_t(img)            # (b,128,H/4,W/4)
        Ic = self.local_encoder_c(canvas)
        Isub = self.local_encoder_d(abs(cha))

        feat = torch.cat([It, Ic, Isub], dim=1)   # (b,384,H/4,W/4)
        feat = self.conv(feat)                    # (b,hidden,H/4,W/4)
        feat = self.downsample(feat)              # (b,hidden,h,w)，h=H/4/2^extra_down
        b, c, h, w = feat.shape

        pos = build_2d_sincos_pos_emb(h, w, c, feat.device)       # (1, h*w, hidden) 全域位置編碼
        memory = feat.flatten(2).permute(0, 2, 1).contiguous() + pos   # (b, h*w, hidden)

        if self.coarse_to_fine:                                   # [B] 尺度純量 → (b,1,1)
            if not torch.is_tensor(scale):
                scale = torch.full((b,), float(scale), device=img.device)
            sc = scale.view(b, 1, 1)                              # 0=細筆，1=粗筆

        anchors01 = None   # [防塌縮] 每個 query 的動態錨點座標（[0,1]），dq_query 時填，供訓練端錨點定位損失
        if self.dq_query:
            # [DQ 動態錨點] 錨點依「當前殘差」每 forward 重新佈，再 grid_sample 該處差異 → query。
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
            grid = anchors.view(b, self.n_queries, 1, 2)         # (b,N,1,2)
            sampled = nn.functional.grid_sample(diff, grid, align_corners=False)  # (b, hidden, N, 1)
            sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()           # (b, N, hidden)
            queries = self.query_norm(self.query_embed[None].expand(b, -1, -1) + sampled)
        else:
            queries = self.query_embed[None].expand(b, -1, -1)   # 純可學習 DETR query

        if self.coarse_to_fine:                                   # [B] 尺度條件加到 query
            queries = queries + self.scale_embed(sc.view(b, 1)).unsqueeze(1)

        hidden_state = self.transformer(memory, queries)         # (b, N, hidden)
        param = self.linear_param(hidden_state)                  # (b, N, param_per_stroke)
        decision = self.linear_decider(hidden_state)             # (b, N, 1)

        # [1B] 有界輸出：形狀參數壓到合法範圍（w/h 恆正、座標 [0,1]），
        # 否則無界線性輸出的負寬高會讓 gaussian_w_distance 出 NaN、訓練發散（DETR head 慣例）。
        P = param.shape[-1]
        if P == 7:   # curved: [x0,y0,x1,y1,x2,y2, w]
            ctrl = torch.sigmoid(param[..., :6])
            width = torch.sigmoid(param[..., 6:7]) * 0.2 + 0.01
            param = torch.cat([ctrl, width], dim=-1)
        else:        # straight: [xc,yc, w(長軸), h(短軸), theta]
            xy = torch.sigmoid(param[..., :2])
            if self.coarse_to_fine:
                # [B] 尺度排程：sc=0 → 細筆做細節（眼/鼻/紋路）；sc=1 → 粗筆鋪底。
                #   長軸 w: [0.01,0.25] → [0.30,0.85]；短軸 h: [0.008,0.08] → [0.20,0.50]。
                #   細端下界調小（原 0.02/0.015）讓模型能下小筆畫細節；上界也收小，
                #   讓細 pass 真的只做精修、不再用中大筆把小特徵蓋掉。下界隨 sc 抬高 → 粗 pass 強制大筆鋪底。
                w_lo = 0.01 + 0.29 * sc; w_hi = 0.25 + 0.60 * sc
                h_lo = 0.008 + 0.192 * sc; h_hi = 0.08 + 0.42 * sc
                w_len = torch.sigmoid(param[..., 2:3]) * (w_hi - w_lo) + w_lo
                h_wid = torch.sigmoid(param[..., 3:4]) * (h_hi - h_lo) + h_lo
            else:
                w_len = torch.sigmoid(param[..., 2:3]) * 0.24 + 0.01    # 長軸 [0.01,0.25]（與 c2f sc=0 一致）
                h_wid = torch.sigmoid(param[..., 3:4]) * 0.072 + 0.008  # 短軸 [0.008,0.08]
            th = torch.sigmoid(param[..., 4:5])
            param = torch.cat([xy, w_len, h_wid, th], dim=-1)

        # 全域採色：用全域座標 param[:,:,:2] 從整張圖 grid_sample。
        s = self.n_queries
        grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()

        return torch.cat([param, color, color, torch.ones(b, s, 1, device=img.device)], dim=-1), decision, anchors01
