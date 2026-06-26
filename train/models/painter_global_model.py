"""[1B] 全域去格畫家模型（單尺度全域 + 回饋精修）。

與原 ``painter_model`` 並存，用 ``--model painter_global`` 切換；原模型完全不動。

核心差異：原版每 32×32 patch 獨立預測 8 筆、座標困在格內 → 網格。
本模型把**整張圖**編碼成 token 序列，用 N 個可學習 query 跨全圖注意力，
輸出**全域座標**筆觸，單一全域畫布合成（無 patch、無棋盤）→ 構造上去格。

繼承 PainterModel 以重用：param2stroke / render_strokes / _apply_real_shapes /
gaussian_w_distance / _shape_w_distance / optimize_parameters（後者本就逐圖做
Hungarian 匹配，只要張量 shape 對就能直接用）。本檔只覆寫 __init__/set_input/forward。

張量形狀（b=batch、N=query 數、M=GT 筆數、R=工作解析度）：
    self.old / render / rec / cha_map : (b, 3, R, R)
    self.gt_param   : (b, M, d_shape)      self.gt_decision   : (b, M)
    self.pred_param : (b, N, d_shape)      self.pred_decision : (b, N)
"""
import os
import torch
import torch.nn.functional as F
import numpy as np

from .painter_model import PainterModel
from . import networks
from . import stroke_render
from . import wgan
from util import morphology


def _dilate(x):
    """[1B 加速] 3x3 dilation = max-pool（與 morphology.Dilation2d 等價，但用融合 kernel 快很多）。"""
    return F.max_pool2d(x, 3, stride=1, padding=1)


def _erode(x):
    """[1B 加速] 3x3 erosion = -max-pool(-x)（與 morphology.Erosion2d 等價）。"""
    return -F.max_pool2d(-x, 3, stride=1, padding=1)


def _composite_over(canvas, fg, alpha):
    """[1B 加速] 向量化 alpha-over：把一個 chunk 的筆觸依序疊到 canvas，用後綴積一次算完，
    取代逐筆 Python 迴圈（從幾十次小 kernel 變幾次大張量運算，GPU 不再被餓著）。

      canvas : (b,3,R,R)             fg, alpha : (b,cs,3,R,R)（依 dim=1 的順序由底到頂疊）
    結果等同 `for i: canvas = fg_i*a_i + canvas*(1-a_i)`。數學：
      out = canvas*∏(1-a) + Σ_i fg_i*a_i*∏_{j>i}(1-a_j)
    """
    om = 1.0 - alpha                                              # (b,cs,3,R,R)
    # C[i] = ∏_{j>=i}(1-a_j)（反向 cumprod）
    C = torch.flip(torch.cumprod(torch.flip(om, dims=[1]), dim=1), dims=[1])
    prod_all = C[:, 0]                                            # ∏ all (1-a)
    ones = torch.ones_like(C[:, :1])
    T = torch.cat([C[:, 1:], ones], dim=1)                       # T[i]=∏_{j>i}(1-a_j)，末筆=1
    contribution = (fg * alpha * T).sum(dim=1)                   # (b,3,R,R)
    return canvas * prod_all + contribution


class PainterGlobalModel(PainterModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 先繼承原版的 real_brush / curved_stroke / lambda_* / used_strokes 等旗標。
        parser = PainterModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataset_mode='null', batch_size=4)  # 全域整張圖，batch 要小
        parser.add_argument('--n_queries', type=int, default=400,
                            help='[1B] 全域 stroke query 數 N（一次能畫的最大筆數）')
        parser.add_argument('--gt_strokes', type=int, default=64,
                            help='[1B] 每張隨機 GT 筆觸數 M（需 <= N）')
        parser.add_argument('--old_strokes', type=int, default=24,
                            help='[1B] 起始畫布 old 的隨機筆觸數')
        parser.add_argument('--global_res', type=int, default=256,
                            help='[1B] 訓練工作解析度 R（推論可更大）')
        parser.add_argument('--global_hidden', type=int, default=256,
                            help='[1B] transformer hidden dim（需可被 4 與 n_heads 整除）')
        parser.add_argument('--extra_down', type=int, default=2,
                            help='[1B] encoder /4 後再 stride2 次數，控制 token 數=(R/4/2^extra_down)^2')
        parser.add_argument('--render_chunk', type=int, default=32,
                            help='[1B] forward 每次渲染的筆觸數；越小越省顯存（morphology 峰值=chunk*batch）')
        parser.add_argument('--dq_query', type=int, default=1,
                            help='[DQ] 1=差異驅動 query（恢復論文 Differential Query），0=純可學習 DETR query')
        parser.add_argument('--coarse_to_fine', type=int, default=0,
                            help='[B] 1=尺度排程（早 step 粗筆鋪底、晚 step 細筆精修，模擬人類作畫），0=單尺度')
        parser.add_argument('--steps', type=int, default=4,
                            help='[路2] 多步漸進訓練的步數 K（在自己畫布上連續補 K 步）')
        parser.add_argument('--target_strokes', type=int, default=256,
                            help='[路2] 合成重建目標的筆觸數（密集度；越大目標越滿）')
        parser.add_argument('--lambda_anchor', type=float, default=4.0,
                            help='[防塌縮] 錨點定位損失權重：逼每個 query 的預測座標靠近自己那個互異的'
                                 '動態錨點，給 400 個 query 各自不同的位置目標 → 打破 pixel loss 的置換對稱、'
                                 '不再塌成同一筆（取代被移除的 Hungarian 指派）。dq_query=0 時自動無效。0=關閉')
        parser.add_argument('--lambda_edge', type=float, default=0.0,
                            help='[細節][實驗，預設關] 邊緣加權 L1：weight=1+λ·|∇target|。本意是放大高對比邊緣的損失逼模型'
                                 '重建眼/鼻，但實測會讓模型在邊緣丟深色雜斑（單色大筆畫不出乾淨輪廓）→ 預設 0 關閉')
        return parser

    def __init__(self, opt):
        # 不呼叫 PainterModel.__init__（它會建 patch 版網路），改自己組裝。
        from .base_model import BaseModel
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'anchor', 'gt', 'w', 'decision', 'decision_sum',
                           'gan', 'D_fake', 'D_real', 'G', 'D']
        self.visual_names = ['old', 'render', 'rec']
        self.model_names = ['g']

        # --- gates（與原版同義）---
        self.curved = bool(getattr(opt, 'curved_stroke', False))    # 1C
        self.real_brush = bool(getattr(opt, 'real_brush', False))   # 3A(+3C)
        self.coarse_to_fine = bool(getattr(opt, 'coarse_to_fine', 0))  # B：尺度排程
        if self.curved:
            self.d, self.d_shape = 14, 7
        else:
            self.d, self.d_shape = 12, 5

        # 全域單張圖計算圖巨大，關閉 retain_graph 避免跨 iteration 累積爆顯存（見 painter_model）。
        self.retain_graph = False
        # WGAN critic 是 32×32 patch 專用，對全域整張圖無意義 → 關閉（之後可換成全圖 discriminator）。
        self.use_critic = False
        # decision_sum 稀疏懲罰按筆數縮放，避免 N=400 時把 decision logit 全壓成 0（見 painter_model）。
        self.decision_sum_coeff = 0.1 * 8.0 / opt.n_queries
        # [防塌縮] 錨點定位損失權重（dq_query 才有錨點可用）。
        self.lambda_anchor = float(getattr(opt, 'lambda_anchor', 4.0))
        # [細節] 邊緣加權 L1 強度（預設 0=純 L1；實測 >0 會在邊緣產生深色雜斑，見 modify_commandline_options）。
        self.lambda_edge = float(getattr(opt, 'lambda_edge', 0.0))

        # --- 全域超參 ---
        self.R = opt.global_res
        self.N = opt.n_queries
        self.M = opt.gt_strokes
        self.M_old = opt.old_strokes
        # [路2] 多步漸進訓練
        self.K = int(getattr(opt, 'steps', 4))
        self.M_target = int(getattr(opt, 'target_strokes', 256))

        # 3A：筆觸庫（內建 2 張或 brush/real/*.png）
        self.meta_brushes = stroke_render.load_meta_brushes(
            getattr(opt, 'brush_dir', 'brush'), self.device, real_brush=self.real_brush)

        # 3C：真實參數池（可選，覆蓋隨機 GT 形狀）
        self.real_params = None
        if self.real_brush:
            rp_path = getattr(opt, 'real_params', 'brush/real_params.npy')
            if rp_path and os.path.isfile(rp_path):
                arr = np.load(rp_path).astype(np.float32)
                if arr.shape[1] == self.d_shape:
                    self.real_params = torch.from_numpy(arr).to(self.device)
                    print('[painter_global] 3C: loaded %d real stroke params' % arr.shape[0])
                else:
                    print('[painter_global] 3C: real_params dim %d != %d，忽略' % (arr.shape[1], self.d_shape))

        # --- 全域 DETR 網路 ---
        net_g = networks.PainterGlobal(
            self.d_shape, self.N, opt.global_hidden, n_heads=8,
            n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks,
            extra_down=opt.extra_down, dq_query=bool(getattr(opt, 'dq_query', 1)),
            coarse_to_fine=self.coarse_to_fine, device=self.device)
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)

        # 狀態
        self.old = self.render = self.rec = self.cha_map = None
        self.gt_param = self.pred_param = None
        self.gt_decision = self.pred_decision = None
        for n in ['pixel', 'anchor', 'gt', 'w', 'decision', 'decision_sum', 'D_fake', 'D_real', 'G', 'D', 'gan']:
            setattr(self, 'loss_' + n, torch.tensor(0., device=self.device))

        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.AdamW(self.net_g.parameters(), lr=opt.lr,
                                               betas=(0.5, 0.999), weight_decay=1e-2)
            self.optimizers.append(self.optimizer)
        self.critic = wgan.Wgan(input_dim=3, dataset_inside=False, device=self.device)

    # ---- 全域筆觸工具 ------------------------------------------------------
    def _rand_global(self, b, n):
        """隨機 n 筆全域筆觸 (b, n, d)；座標相對**整張畫布** [0,1]。

        全域筆觸尺寸應遠小於整張畫布（不像 patch 版佔半個 32px），故 w,h/寬偏小。
        """
        p = torch.rand(b, n, self.d, device=self.device)
        if self.curved:
            p[:, :, :6] = p[:, :, :6] * 0.9 + 0.05   # 3 控制點散布全圖 [0.05,0.95]
            p[:, :, 6] = p[:, :, 6] * 0.10 + 0.02    # 寬 [0.02,0.12]（相對整張）
        else:
            p[:, :, 0:2] = p[:, :, 0:2] * 0.9 + 0.05   # xc,yc 全圖 [0.05,0.95]
            # [路A] 多樣化形狀：長軸 w 從短到長、短軸 h 維持細 → 各種長寬比（含長條斜筆），
            # 配隨機 θ 涵蓋各角度。模型因此學會「依位置挑不同形狀的筆」（如原論文的多形狀筆觸），
            # 而非只會方塊。注意是「隨機但多樣」，不是全部變長。
            p[:, :, 2] = p[:, :, 2] * 0.44 + 0.01      # w 長軸 [0.01,0.45]（含細筆，讓 GT 有細節可學）
            p[:, :, 3] = p[:, :, 3] * 0.112 + 0.008    # h 短軸 [0.008,0.12]
        p[:, :, -4:-1] = p[:, :, -7:-4]               # 頭尾色一致
        return p

    def _render_set(self, param_bnd):
        """把 (b, n, d) 一批筆觸渲染成 (b, n, 3, R, R) 前景與 alpha。"""
        b, n, d = param_bnd.shape
        fg, al = self.render_strokes(param_bnd.reshape(-1, d).contiguous(), self.R, self.R)
        fg = _dilate(fg)
        al = _erode(al)
        fg = fg.view(b, n, 3, self.R, self.R)
        al = al.view(b, n, 3, self.R, self.R)
        return fg, al

    # ---- 路2：多步漸進訓練 -------------------------------------------------
    @torch.no_grad()
    def _make_target(self, b):
        """合成一張密集的「油畫」當重建目標：M_target 筆隨機筆觸分塊合成到空白。"""
        canvas = torch.zeros(b, 3, self.R, self.R, device=self.device)
        params = self._apply_real_shapes(self._rand_global(b, self.M_target))  # 3C（無 real_params 即 no-op）
        chunk = getattr(self.opt, 'render_chunk', 32)
        for s in range(0, self.M_target, chunk):
            e = min(s + chunk, self.M_target)
            sub = params[:, s:e].reshape(-1, self.d).contiguous()
            fg, al = self.render_strokes(sub, self.R, self.R)
            fg = _dilate(fg); al = _erode(al)
            cs = e - s
            fg = fg.view(b, cs, 3, self.R, self.R)
            al = al.view(b, cs, 3, self.R, self.R)
            canvas = _composite_over(canvas, fg, al)            # 向量化合成（無 dec，全保留）
        return canvas

    def _step_scale(self, k):
        """[B] 第 k 步（共 K 步）的尺度：step0 最粗(1.0) → 末步最細(0.0)，模擬先鋪底再精修。
        未開 --coarse_to_fine 時恆 0（單尺度，與原行為一致）。"""
        if not self.coarse_to_fine or self.K <= 1:
            return 0.0
        return 1.0 - k / (self.K - 1)

    def _pyramid_res(self):
        """[金字塔] K 步的解析度排程：低→高（R=256,K=4 → 64,64,128,256；K=3 → 64,128,256）。
        複製原 patch 版金字塔的準確機制——低解析度層大筆粗鋪、高解析度層看細節下小筆——
        但**每層全域**（整張圖一次 forward、不切 patch → 去格保住）。下界 64 避免特徵圖 token 太少。
        未開 coarse_to_fine 或 K<=1 時全在 R（與單尺度行為一致）。"""
        if not self.coarse_to_fine or self.K <= 1:
            return [self.R] * self.K
        return [max(self.R // (2 ** (self.K - 1 - k)), 64) for k in range(self.K)]

    def _paint_step(self, target, canvas_in, scale=0.0):
        """一步：net 看 (target, canvas_in, cha) → 預測 N 筆、分塊合成到 canvas_in。
        canvas_in 須已 detach（每步獨立）。回傳 (canvas_out, decision_logits (b,N), loss_anchor)。
        **渲染解析度取 canvas_in 的空間大小**（金字塔每層 res 不同；筆觸是正規化座標，任意 res 皆可渲染）。

        loss_anchor = |pred_xy − 自己的動態錨點|（dq_query 才有；否則 0）。它給每個 query
        一個互異的位置目標，打破 pixel loss 對 400 個 query 的置換對稱，避免 decoder 把所有
        query 塌成同一筆（取代路2 移除掉的 Hungarian 指派）。"""
        b, R = canvas_in.shape[0], canvas_in.shape[-1]
        cha = target - canvas_in
        param, decisions, anchors01 = self.net_g(target, canvas_in, cha, scale=scale)
        # [防塌縮] 錨點定位損失：pred 的前兩維 (xc,yc) 對齊各自錨點（anchors01 已 detach，當位置目標）。
        if anchors01 is not None and self.lambda_anchor > 0:
            loss_anchor = (param[:, :, :2] - anchors01).abs().mean()
        else:
            loss_anchor = torch.zeros((), device=canvas_in.device)
        # [路2] 軟決策：用 sigmoid 當每筆不透明度（連續、梯度平滑），避免硬閾值 + 稀疏懲罰把 decision 塌縮成全負。
        dec = torch.sigmoid(decisions.view(b, self.N, 1, 1, 1).contiguous())
        chunk = getattr(self.opt, 'render_chunk', 32)
        canvas = canvas_in
        for s in range(0, self.N, chunk):
            e = min(s + chunk, self.N)
            sub = param[:, s:e].reshape(-1, self.d).contiguous()
            fg, al = self.render_strokes(sub, R, R)
            fg = _dilate(fg); al = _erode(al)
            cs = e - s
            fg = fg.view(b, cs, 3, R, R)
            al = al.view(b, cs, 3, R, R)
            alpha = al * dec[:, s:e]                             # (b,cs,3,R,R)，含 decision 閘
            canvas = _composite_over(canvas, fg, alpha)         # 向量化合成
        return canvas, decisions.view(b, self.N), loss_anchor

    @torch.no_grad()
    def _edge_weight(self, target):
        """[細節] 由目標圖梯度幅值算逐像素 L1 權重 (b,1,R,R)：weight = 1 + lambda_edge·|∇target|（每張正規化到[0,1]）。
        高對比邊緣（眼/鼻/輪廓）權重大 → pixel loss 在那裡被放大、模型有動力重建。lambda_edge=0 → 全 1（純 L1）。"""
        if self.lambda_edge <= 0:
            return torch.ones_like(target[:, :1])
        gray = target.mean(1, keepdim=True)                          # (b,1,R,R)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                          device=target.device).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        gx = F.conv2d(gray, kx, padding=1); gy = F.conv2d(gray, ky, padding=1)
        e = (gx * gx + gy * gy).sqrt()
        e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-6)            # 每張正規化到 [0,1]
        return 1.0 + self.lambda_edge * e

    def set_input(self, input_dict):
        self.image_paths = input_dict['A_paths']
        b = self.opt.batch_size
        with torch.no_grad():
            self.target = self._make_target(b)              # 密集重建目標
            self.edge_w = self._edge_weight(self.target)    # [細節] 邊緣加權圖（pixel loss 用）
            self.old = torch.zeros_like(self.target)        # 空白起步（visualizer 顯示）
            self.render = self.target                       # visualizer 顯示目標
            self.cha_map = self.target - self.old

    def forward(self):
        """推論/視覺化：金字塔逐層升解析度跑 K 層無梯度，self.rec = 最終畫布（升回 R）。"""
        with torch.no_grad():
            res_list = self._pyramid_res()
            b = self.target.shape[0]
            canvas = torch.zeros(b, 3, res_list[0], res_list[0], device=self.device)
            for k, res in enumerate(res_list):
                target_L = F.interpolate(self.target, (res, res), mode='area')
                if canvas.shape[-1] != res:
                    canvas = F.interpolate(canvas, (res, res), mode='bilinear', align_corners=False)
                canvas, _, _ = self._paint_step(target_L, canvas, scale=self._step_scale(k))
            self.rec = canvas if canvas.shape[-1] == self.R else \
                F.interpolate(canvas, (self.R, self.R), mode='bilinear', align_corners=False)

    def optimize_parameters(self, epoch):
        """[金字塔 + 路2] 低解析度空白起步，逐層升 res，在自己的(detached)畫布上補 K 層，
        每層 pixel loss 拉近「該層解析度的目標」、立即 backward（記憶體=單層；低層便宜、峰值=最高層）。
        模型因此學會『先在低 res 粗鋪、升 res 後在殘差上補細節』——複製原 patch 版金字塔準確度，但每層全域。"""
        self.optimizer.zero_grad()
        res_list = self._pyramid_res()
        b = self.target.shape[0]
        canvas = torch.zeros(b, 3, res_list[0], res_list[0], device=self.device)
        last_pixel = torch.tensor(0., device=self.device)
        last_anchor = torch.tensor(0., device=self.device)
        for k, res in enumerate(res_list):
            target_L = F.interpolate(self.target, (res, res), mode='area')      # 該層解析度的目標
            if canvas.shape[-1] != res:
                canvas = F.interpolate(canvas, (res, res), mode='bilinear', align_corners=False)  # 上層結果升採樣帶上來
            canvas_in = canvas.detach()                     # 每層以 detach 畫布為輸入（單層圖）
            canvas, _, loss_anchor = self._paint_step(target_L, canvas_in, scale=self._step_scale(k))
            # pixel 重建（軟決策不透明度由 pixel loss 自然學）+ 錨點定位損失（防 query 置換對稱塌縮）。
            if self.lambda_edge > 0:
                w = F.interpolate(self.edge_w, (res, res), mode='area')
                loss_pixel = ((canvas - target_L).abs() * w).mean()
            else:
                loss_pixel = (canvas - target_L).abs().mean()
            loss_k = loss_pixel * self.opt.lambda_pixel + loss_anchor * self.lambda_anchor
            loss_k.backward()                               # 立即 backward 釋放本層圖
            last_pixel = loss_pixel.detach()
            last_anchor = loss_anchor.detach()
            canvas = canvas.detach()
        self.optimizer.step()
        self.rec = canvas
        # 損失記錄（路2 用 pixel + anchor；其餘軸設 0）
        self.loss_pixel = last_pixel
        self.loss_anchor = last_anchor
        self.loss_G = last_pixel * self.opt.lambda_pixel + last_anchor * self.lambda_anchor
        for n in ['gt', 'w', 'decision', 'decision_sum', 'gan', 'D_fake', 'D_real', 'D']:
            setattr(self, 'loss_' + n, torch.tensor(0., device=self.device))
