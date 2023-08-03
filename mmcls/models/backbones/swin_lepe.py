# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint

from mmcls.utils import get_root_logger
from ..utils import  to_2tuple
from ..builder import BACKBONES
from ..utils import DetPatchEmbed as PatchEmbed
from ..utils import DetPatchMerging as PatchMerging 
from ..utils import swin_converter
from .pvt import PVTEncoderLayer, MixFFN
from .cswin_transformer import CSWinBlock

class WindowMSALePE(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with locally-enhanced
       position encoding (from CSWin).

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 use_mixffn=False, # if set use_mixffn to True. will shutdown lepe
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.use_mixffn = use_mixffn
        self.init_cfg = init_cfg

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        if not self.use_mixffn:
            self.get_v = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, groups=embed_dims)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        pass

    def get_rpe(self, x, func):
        """input:  gird tokoes x
                 :    B' num_heads L' C' 
           output:    B' num_heads L' C'  
        """
        B, num_heads, L, C = x.shape
        Wh, Ww = self.window_size
        assert L == Wh*Ww
        x = x.permute(0, 1, 3, 2).reshape(B, num_heads*C, L).view(B, num_heads*C, Wh, Ww).contiguous() ## B' C H' W', C = C'*num_heads

        rpe = func(x) ### B', C, H', W'
        
        rpe = rpe.reshape(B, num_heads, C , Wh * Ww).permute(0, 1, 3, 2).contiguous() ## B num_heads L C
        return rpe

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4) 
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2] # B n_heads N C
        if not self.use_mixffn:
            rpe = self.get_rpe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        if not self.use_mixffn:
            x = (attn @ v) + rpe
        else:
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSALePE(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 use_mixffn=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSALePE(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            use_mixffn=use_mixffn,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinLePEBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 parallel_group=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 use_mixffn=False,
                 init_cfg=None):

        super(SwinLePEBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.parallel_group=parallel_group
        self.use_mixffn = use_mixffn

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.branch_num = 2 if self.parallel_group else 1
        shift = True if self.parallel_group else shift

        self.attns = ModuleList([ShiftWindowMSALePE(
            embed_dims=embed_dims//self.branch_num,
            num_heads=num_heads//self.branch_num,
            window_size=window_size,
            shift_size=window_size // 2 if (shift & (not i)) else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            use_mixffn=use_mixffn,
            init_cfg=None) for i in range(self.branch_num)])

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        if not use_mixffn:
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                add_identity=True,
                init_cfg=None)
        else:
            self.mixffn = MixFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                use_conv=True, # set to True to use pvtv2
                act_cfg=act_cfg)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            B, L, C = x.shape
            if self.branch_num == 2:
                x1 = self.attns[0](x[:,:,:C//2], hw_shape)
                x2 = self.attns[1](x[:,:,C//2:], hw_shape)
                x = torch.cat([x1,x2], dim=2)
            else:
                x = self.attns[0](x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            if not self.use_mixffn:
                x = self.ffn(x, identity=identity)
            else:
                x = self.mixffn(x, hw_shape, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinLePEBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 parallel_group=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 use_patchemb_downsample=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 use_mixffn=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.use_patchemb_downsample=use_patchemb_downsample

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinLePEBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                parallel_group=parallel_group,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                use_mixffn=use_mixffn,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            if not self.use_patchemb_downsample:
                x_down, down_hw_shape = self.downsample(x, hw_shape)
            else:
                B, L, C = x.shape
                H, W = hw_shape
                x = x.view(B, H, W, C).permute([0, 3, 1, 2])
                x_down, down_hw_shape = self.downsample(x)
                x = x.permute(0, 2, 3, 1).view(B, H*W, C)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class SwinLePEPVTBlockSequence(BaseModule):
    """Implements one stage of the PVT & SwinLePE Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 parallel_group=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 #==pvt configs==
                 pvt_pos='rear',
                 use_conv_ffn=True, # PVT-v2 
                 sr_ratio=1,
                 use_cswin=False,
                 #======end======
                 swin_block_num=0,
                 use_global_shift=False,
                 use_patchemb_downsample=False, 
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 use_mixffn=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.use_patchemb_downsample = use_patchemb_downsample
        self.swin_block_num = swin_block_num
        self.use_cswin = use_cswin
        self.blocks = ModuleList()
        assert pvt_pos in ['rear', 'front', 'both']
        if pvt_pos == 'rear':
            target_layer = [depth-1]
        elif pvt_pos == 'front':
            target_layer = [0]
        else:
            target_layer = [0, depth-1]
        for i in range(depth):
            if i in target_layer:
                block = PVTEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratio,
                    use_conv_ffn=use_conv_ffn,
                    init_cfg=None)
            else:
                if use_global_shift:
                    shift_flag = False if self.swin_block_num % 2 == 0 else True
                else:
                    shift_flag = False if i % 2 == 0 else True
                block = SwinLePEBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=feedforward_channels,
                    window_size=window_size,
                    parallel_group=parallel_group,
                    shift=shift_flag,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    use_mixffn=use_mixffn,
                    init_cfg=None)
                self.swin_block_num += 1
            self.blocks.append(block)

        if self.use_cswin:
            for i in target_layer:
                self.blocks[i] = CSWinBlock(
                        dim=embed_dims,
                        num_heads=num_heads,
                        patches_resolution=224//32,
                        mlp_ratio=feedforward_channels//embed_dims,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        split_size=1,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_rates[i],
                        fix_grid_size=False,
                        norm_layer=nn.LayerNorm,
                        last_stage=True,
                        init_cfg=None)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            if not self.use_patchemb_downsample:
                x_down, down_hw_shape = self.downsample(x, hw_shape)
            else:
                B, L, C = x.shape
                H, W = hw_shape
                x = x.view(B, H, W, C).permute([0, 3, 1, 2])
                x_down, down_hw_shape = self.downsample(x)
                x = x.permute(0, 2, 3, 1).view(B, H*W, C)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@BACKBONES.register_module()
class SwinLePETransformer(BaseModule):
    """ SwinLePE Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030
    with modified Le Position Encoding

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        mixed_arch (bool): If True, use mixed arch of PVT and SwinLePE.
            Defaults: False
        pvt_pos (str): The position of PVT block in each stage. Defaults: 'rear'.
            Could be set to 'rear', 'front' or 'both'
        use_conv_ffn (bool): FFN in PVT. Defaults: True. The same as PVT-v2
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 paddings=[0, 0, 0, 0],    # used for pathembed pooling
                 patch_sizes=[4, 2, 2, 2], # used for pathembed pooling
                 out_indices=(0, 1, 2, 3),
                 parallel_group=False,
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 #==pvt configs==
                 mixed_arch=False,
                 pvt_stage=[True, True, True, True],
                 pvt_pos='rear',
                 use_conv_ffn=True, # PVT-v2 
                 sr_ratios=[8, 4, 2, 1],
                 use_cswin=False,
                 #======end======
                 use_mixffn=False, # Set to True to replace lepe to mixffn
                 use_patchemb_downsample=False,
                 use_global_shift=False,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinLePETransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.use_global_shift = use_global_shift
        self.use_patchemb_downsample = use_patchemb_downsample
        self.total_swin_block_num = 0

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                if not self.use_patchemb_downsample:
                    downsample = PatchMerging(
                        in_channels=in_channels,
                        out_channels=2 * in_channels,
                        stride=strides[i + 1],
                        norm_cfg=norm_cfg if patch_norm else None,
                        init_cfg=None)
                else:
                    downsample = PatchEmbed(
                        in_channels=in_channels,
                        embed_dims=2 * in_channels,
                        conv_type='Conv2d',
                        kernel_size=patch_sizes[i + 1],
                        stride=strides[i + 1],
                        padding=paddings[i + 1],
                        norm_cfg=norm_cfg if patch_norm else None,
                        init_cfg=None)
            else:
                downsample = None
            
            if mixed_arch and pvt_stage[i]: # add pvt block to SwinLePE 
                stage = SwinLePEPVTBlockSequence(
                    embed_dims=in_channels,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * in_channels,
                    depth=depths[i],
                    window_size=window_size,
                    parallel_group=parallel_group,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    downsample=downsample,
                    use_patchemb_downsample=self.use_patchemb_downsample,
                    pvt_pos=pvt_pos,
                    use_mixffn=use_mixffn,
                    use_cswin=use_cswin,
                    use_global_shift=self.use_global_shift,
                    swin_block_num=self.total_swin_block_num,
                    use_conv_ffn=use_conv_ffn,
                    sr_ratio=sr_ratios[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    init_cfg=None)
                self.total_swin_block_num = stage.swin_block_num
            else:                   
                stage = SwinLePEBlockSequence(
                    embed_dims=in_channels,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * in_channels,
                    depth=depths[i],
                    window_size=window_size,
                    parallel_group=parallel_group,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    use_mixffn=use_mixffn,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    use_patchemb_downsample=self.use_patchemb_downsample,
                    downsample=downsample,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    init_cfg=None)            
            self.stages.append(stage)
            if downsample:
                if not self.use_patchemb_downsample:
                    in_channels = downsample.out_channels
                else:
                    in_channels = downsample.embed_dims

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinLePETransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                logger.warning('Loading weights from original repo with convert_weights mode. \
                                If mm-based weights are used, please restart and set convert_weigths to False.')
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # load state_dict
            msg = self.load_state_dict(state_dict, False)
            logger.info(msg)
            del ckpt 
            torch.cuda.empty_cache()

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return tuple(outs)
