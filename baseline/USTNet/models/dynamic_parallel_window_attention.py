import torch
import torch.nn as nn
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple, trunc_normal_

"""
基于Swin Transformer架构，引入并行的双流注意力机制
depths: 每个stage中Transformer block的数量列表，这里只有1个stage包含12个block
（Stage 指的是网络中的一个处理阶段，通常具有相同的特征维度和分辨率。在层次化Transformer中（如Swin Transformer），
网络被分成多个stage，每个stage包含：相同的特征维度，相同的空间分辨率，多个相同配置的Transformer block）
num_heads: 每个stage中注意力头的数量列表，这里为12个头
embed_dim: 基础特征维度，输入输出的通道数; window_size: 局部注意力窗口的大小; mlp_ratio: MLP隐藏层维度与嵌入维度的比例（384×4=1536）
qkv_bias: 是否在QKV线性变换中使用偏置项；qk_scale: QK点积的缩放因子，None表示使用默认的head_dim^-0.5
drop_rate: 普通dropout率; attn_drop_rate: 注意力dropout率
drop_path_rate: 随机深度（Stochastic Depth）的最大概率; norm_layer=nn.LayerNorm: 归一化层类型
"""
class DynamicParallelWindowAttention(nn.Module):
    def __init__(self, depths=[12], num_heads=[12], embed_dim=384,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_layers = len(depths)       # 1个stage，包含12个swin transformer block
        self.embed_dim = embed_dim          # 嵌入维度，输入输出的通道数，384
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))     # 最终特征维度，384
        self.mlp_ratio = mlp_ratio          # MLP扩展比例，4

        self.pos_drop = nn.Dropout(p=drop_rate)         # 位置编码dropout

        # stochastic depth：随机深度衰减规则，生成drop path概率列表，用于每个block的随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers：构建网络层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),               # 当前层维度，384*(2^0)
                               depth=depths[i_layer],                           # 当前stage中Transformer block的数量，12
                               num_heads=num_heads[i_layer],                    # 当前stage中注意力头的数量，12
                               window_size=window_size,                         # 窗口大小，4
                               shift_size=window_size // 2,                     # 窗口滑动大小，2（SW-MSA用）
                               mlp_ratio=self.mlp_ratio,                        # MLP隐藏层维度与嵌入维度的比例，4
                               qkv_bias=qkv_bias, qk_scale=qk_scale,            # QKV线性变换是否使用偏置项，None表示使用默认的head_dim^-0.5
                               drop=drop_rate, attn_drop=attn_drop_rate,        # 普通dropout率，0.0；注意力dropout率，0.0
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],      # 当前stage中随机深度概率列表，drop path概率
                               norm_layer=norm_layer)                           # layerNorm
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)           # 最终归一化层

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}                       # 忽略绝对位置编码的权重衰减

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}             # 忽略相对位置偏置表的权重衰减

    # 空间展开 → 位置dropout → 多层处理 → 归一化 → 空间还原
    def forward_features(self, x):
        b, c, h, w = x.shape                    # 输入: x shape = [b, c, h, w] = [nb, 384, h/16, w/16]
        x = x.flatten(2).transpose(1, 2)        # 将空间维度展平: [nb, 384, h/16, w/16] -> [nb, h/16*w/16, 384]
        x = self.pos_drop(x)                    # 应用位置编码dropout

        for layer in self.layers:
            x = layer(x, h, w)                  # 逐层处理，传递当前特征图和空间维度

        x = self.norm(x)                        # 最终归一化，shape = [nb, h/16*w/16, 384]
        x = x.transpose(1, 2).reshape(b, c, h, w)  # 恢复空间维度: [nb, h/16*w/16, 384] -> [nb, 384, h/16, w/16]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


"""
包含多个DynamicParallelWindowAttentionBlock的容器层, 通过alternate参数实现动态切换机制（alternate=0，偶数block）
"""
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim          # 输入通道数，384
        # self.input_resolution = input_resolution
        self.depth = depth      # 当前stage中Transformer block的数量，12

        # build blocks， 构建DPWA block序列
        self.blocks = nn.ModuleList([
            DynamicParallelWindowAttentionBlock(dim=dim,                                    # 输入通道数，384
                                                num_heads=num_heads,                        # 当前stage中注意力头的数量，12
                                                window_size=window_size,                    # 窗口大小，4
                                                shift_size=shift_size,                      # 窗口滑动大小，2（SW-MSA用）
                                                mlp_ratio=mlp_ratio,                        # MLP隐藏层维度与嵌入维度的比例，4
                                                alternate=0 if (i % 2 == 0) else 1,         # 交替模式，0或1
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,       # QKV线性变换是否使用偏置项，None表示使用默认的head_dim^-0.5
                                                drop=drop, attn_drop=attn_drop,             # 普通dropout率，0.0；注意力dropout率，0.0
                                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,   # 当前block的随机深度概率，drop path概率
                                                norm_layer=norm_layer)                      # 归一化层，LayerNorm
            for i in range(depth)])     # 构建depth个DPWA block

    def forward(self, x, h, w):
        for blk in self.blocks:
            x = blk(x, h, w)            # [b, h*w, c] -> [b, h*w, c]
        return x


"""
通过alternate参数实现动态切换机制（alternate=0，偶数block）
将384维特征分成两个192维的流，每个流使用6个注意力头（6个头并行处理）
"""
class DynamicParallelWindowAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, shift_size=2, alternate=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.alternate = alternate              # 动态切换的关键

        # 两个并行的窗口注意力模块，每个处理一半的通道和一半的注意力头
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),                # 第一个注意力流
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),                # 第二个注意力流
        ])

        # ---- 新增：gate fuse，用于对两路输出按 token 学习加权 ----
        # 输入是 dim，输出是标量 gate per token -> sigmoid -> [0,1]
        fuse_hidden = max(8, dim // 4)
        self.fuse_gate = nn.Sequential(
            nn.Linear(dim, fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fuse_hidden, 1)
        )
        # --------------------------------------------------------

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, h, w):
        H = h
        W = w
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        attn_mask1 = None
        attn_mask2 = None

        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))                        # 创建划分9个区域的掩码, 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)                  # 用于SW-MSA的掩码
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0)).to(
                x.device)

        shortcut = x
        x = self.norm1(x)

        # double attn
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)      # 生成QKV
        if self.alternate == 0:
            qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)          # 前192维，使用常规W-MSA

            if self.shift_size > 0:
                qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2)).reshape(3, B, H, W, C // 2)     # 后192维，用SW-MSA
            else:
                qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)
        else:
            # 动态切换，交换两个流的角色
            qkv_1 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)           # 后192维，用用常规W-MSA

            if self.shift_size > 0:
                qkv_2 = torch.roll(qkv[:, :, :, :C // 2], shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2)).reshape(3, B, H, W, C // 2)     # 前192维，用SW-MSA
            else:
                qkv_2 = qkv[:, :, :, :C // 2].reshape(3, B, H, W, C // 2)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)         # 窗口划分
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, attn_mask1)       # 并行注意力计算，无掩码
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, attn_mask2)

        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)       # 还原到原始空间布局
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))         # 移位还原
        else:
            x2 = x2


        # if self.alternate == 0:
        #     x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)      # 特征拼接
        # else:
        #     x = torch.cat([x2.reshape(B, H * W, C // 2), x1.reshape(B, H * W, C // 2)], dim=2)
        #
        # x = self.proj(x)                                    # 投影层

        # reshape为 (B, L, C//2)
        x1_r = x1.reshape(B, H * W, C // 2)
        x2_r = x2.reshape(B, H * W, C // 2)

        # ---- 使用 gate 对两路按 token 加权，再 concat（保持原有 alternate 顺序） ----
        g = torch.sigmoid(self.fuse_gate(x))                                # [B, L, 1]
        if self.alternate == 0:
            left = g * x1_r
            right = (1.0 - g) * x2_r
            x = torch.cat([left, right], dim=2)                     # [B, L, C]
        else:
            left = g * x2_r
            right = (1.0 - g) * x1_r
            x = torch.cat([left, right], dim=2)
        x = self.proj(x)
        # --------------------------------------------------------

        # FFN
        x = shortcut + self.drop_path(x)                    # 第一次残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # MLP+第二次残差连接

        return x

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        # print(B_, N, C)
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # print(mask.size())
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





