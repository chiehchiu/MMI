# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint

from mmcls.utils import get_root_logger
from ..utils import  to_2tuple
from ..builder import BACKBONES
from ..utils import PatchEmbed3D, PatchMerging3D, swin_converter


class WindowMSA3D(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The depth, height and width of the window.
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
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  #2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wd, Wh, Ww = self.window_size
        """
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        """
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        rel_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

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
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(
                N, N, -1)
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        """
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA3D(BaseModule):
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
                 window_size=(2,7,7),
                 shift_size=(0,0,0),
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        if isinstance(window_size, int) or isinstance(window_size, float):
            window_size = (window_size, ) * 3
        self.window_size = window_size
        if isinstance(shift_size, int) or isinstance(shift_size, float):
            shift_size = (shift_size,) * 3
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.w_msa = WindowMSA3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, dhw_shape):
        B, L, C = query.shape
        D, H, W = dhw_shape
        assert L == D * H * W, 'input feature has wrong size'
        query = query.view(B, D, H, W, C)

        # pad feature maps to multiples of window size
        H_pad, W_pad = query.shape[1], query.shape[2]

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        query = F.pad(query, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, D_pad, H_pad, W_pad, _ = query.shape

        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size[2], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3))

            # calculate attention mask for SW-MSA-3D
            img_mask = torch.zeros((1, D_pad, H_pad, W_pad, 1), device=query.device)
            d_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0],
                              -self.shift_size[0]), slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1],
                              -self.shift_size[1]), slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2],
                              -self.shift_size[2]), slice(-self.shift_size[2], None))
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, reduce(mul, self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size*window_size, C
        query_windows = query_windows.view(-1, reduce(mul, self.window_size), C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0],
                                             self.window_size[1],
                                             self.window_size[2], C)

        # B D' H' W' C
        shifted_x = self.window_reverse(attn_windows, D_pad, H_pad, W_pad)
        # reverse cyclic shift
        if any(i > 0 for i in self.shift_size): 
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = x.view(B, D * H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, D, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, window_size, C)
            D (int): Depth of image
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, D, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W * D/ window_size[0] / window_size[1] / window_size[2]))
        x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], 
                                 window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, D, H, W, C)
        Returns:
            windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
        """
        B, D, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, D // window_size[0], window_size[0],
                      H // window_size[1], window_size[1], 
                      W // window_size[2], window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        return windows


class SwinBlock(BaseModule):
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
                 window_size=(2, 7, 7),
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=tuple(np.array(window_size) // 2) if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, dhw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, dhw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence3D(BaseModule):
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
                 window_size=(2, 7, 7),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        if isinstance(window_size, int) or isinstance(window_size, float):
            window_size = (window_size, ) * 3

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, dhw_shape):
        for block in self.blocks:
            x = block(x, dhw_shape)

        if self.downsample:
            x_down, down_dhw_shape = self.downsample(x, dhw_shape)
            return x_down, down_dhw_shape, x, dhw_shape
        else:
            return x, dhw_shape, x, dhw_shape


@BACKBONES.register_module()
class SwinTransformer3D(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (tuple[int]): Patch size. Default: (1, 4, 4). 
        window_size (tuple): Window size. Default: (2, 7, 7).
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
                 patch_size=(4, 4, 4),
                 window_size=(2, 7, 7),
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=((1,4,4), 2, 2, 2), # 2 equals to (1,2,2),use (2,2,2) if pooling on all dims are needed
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 pretrained2d=False, # the pre-trained weights are 2D or not (3D).
                 loaded_window_size=(2, 7, 7),  # to interpolate 3D realative bias table.
                 loaded_strides=(1,2,2), # needed for patch merging weights inflate
                 loaded_stem_strides=(1,4,4), # needed for patch embeding weights inflate
                 output_2d=False,
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

        if isinstance(window_size, int) or isinstance(window_size, float):
            window_size = (window_size, ) * 3
        self.window_size = window_size
        assert len(loaded_window_size)==3
        self.loaded_window_size = loaded_window_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.output_2d = output_2d
        self.loaded_strides = loaded_strides
        self.strides = strides

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

        super(SwinTransformer3D, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.pretrained2d = pretrained2d

        assert len(patch_size) == 3
        if not strides[0][2] == patch_size[2]:
            warnings.warn("Use non-overlapping patch embed")

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv3d',
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
                downsample = PatchMerging3D(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence3D(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
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
                                This impl does not support mm-based 2d swin backbone weights, please use original repo weights instead!') 
                if self.pretrained2d:
                    logger.warning("3D Swin use original impl for patch merging. Therefore patch merging weights could be directly loaded!")
                    logger.warning("Only Support xy-axis patch merging (1, 2, 2) in 3d from 2d mode, and patch embeding (any,4,4)")
                    _state_dict = swin_converter(_state_dict, self.pretrained2d)
                else:
                    logger.warning("Convert weights are only implemented for 2d original repo weights")
                    logger.warning("Set convert_weights to False if mm pretrained 2D weights are used")
                    raise NotImplementedError

            # delete relative_position_index since we always re-init it
            relative_position_index_keys = [k for k in _state_dict.keys() if "relative_position_index" in k]
            for k in relative_position_index_keys:
                del _state_dict[k]
    
            # delete attn_mask since we always re-init it
            attn_mask_keys = [k for k in _state_dict.keys() if "attn_mask" in k]
            for k in attn_mask_keys:
                del _state_dict[k]

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

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            if self.pretrained2d:
                """Inflate the swin2d parameters to swin3d.
            
                The differences between swin3d and swin2d mainly lie in an extra
                axis. To utilize the pretrained parameters in 2d model,
                the weight of swin2d models should be inflated to fit in the shapes of
                the 3d counterpart.
                """
                state_dict['patch_embed.projection.weight'] = state_dict['patch_embed.projection.weight'].unsqueeze(2).repeat(1,1, self.patch_size[0],1,1) / self.patch_size[0]
                if self.in_channels == 1:
                    state_dict['patch_embed.projection.weight'] = torch.sum(state_dict['patch_embed.projection.weight'],1).unsqueeze(1)

                for table_key in relative_position_bias_table_keys:
                    table_pretrained = state_dict[table_key]
                    table_current = self.state_dict()[table_key]
                    L1, nH1 = table_pretrained.size()
                    L2, nH2 = table_current.size()
                    L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
                    wd = self.window_size[0]
                    if nH1 != nH2:
                        logger.warning(f'Error in loading {table_key}, pass')
                    elif L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        table_pretrained = table_pretrained_resized
                    wd = self.window_size[0]
                    state_dict[table_key] = table_pretrained.view(
                           nH2, L2).permute(1, 0).contiguous().repeat(2*wd-1,1)
            else:
                # inflate 3D weights on the d-dims if needed. (when pretrained with VDT, we use 1*4*4 patches sometime.)
                if state_dict['patch_embed.projection.weight'].shape[2] != self.patch_size[0]:
                    assert state_dict['patch_embed.projection.weight'].shape[2] == 1
                    state_dict['patch_embed.projection.weight'] = state_dict['patch_embed.projection.weight'].repeat(1,1, self.patch_size[0],1,1) / self.patch_size[0]
                # if 3D pre-trained weights are used
                for table_key in relative_position_bias_table_keys:
                    table_pretrained = state_dict[table_key]
                    table_current = self.state_dict()[table_key]
                    L1, nH1 = table_pretrained.size()
                    assert L1 == (2*self.loaded_window_size[1]-1) * (2*self.loaded_window_size[2]-1) * (2*self.loaded_window_size[0]-1)
                    L2, nH2 = table_current.size()
                    assert L2 == (2*self.window_size[1]-1) * (2*self.window_size[2]-1) * (2*self.window_size[0]-1)
                    wd = self.window_size[0]
                    if nH1 != nH2:
                        logger.warning(f'Error in loading {table_key}, pass')
                    elif L1 != L2:
                        d1, h1, w1 = self.loaded_window_size
                        d2, h2, w2 = self.window_size
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1, 0).reshape(1, nH1, 2*d1-1, 2*h1-1, 2*w1-1),
                            size=(2*d2-1, 2*h2-1, 2*w2-1),
                            mode='trilinear')
                        table_pretrained = table_pretrained_resized
                    state_dict[table_key] = table_pretrained.view(
                           nH2, L2).permute(1, 0).contiguous()
                # Inflate patch merging weights if needed. Only support from 1,2,2, to 2,2,2.
                downsample_reduction_keys = [
                    k for k in state_dict.keys()
                    if 'downsample.reduction' in k
                ]
                downsample_norm_keys = [
                    k for k in state_dict.keys()
                    if 'downsample.norm' in k
                ]
                if self.loaded_strides==(1,2,2) and self.strides[1]==(2,2,2):
                    logger.warning("Inflate Patch Merging Reduction weights")
                    for red_key in downsample_reduction_keys:
                        state_dict[red_key] = state_dict[red_key].repeat(1,2)/2.
                    for norm_key in downsample_norm_keys:
                        state_dict[norm_key] = state_dict[norm_key].repeat(2)

            # load state_dict
            msg = self.load_state_dict(state_dict, False)
            logger.info(msg)
            del ckpt
            torch.cuda.empty_cache()

    def forward(self, x):
        x, dhw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, dhw_shape, out, out_dhw_shape = stage(x, dhw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_dhw_shape,
                               self.num_features[i]).permute(0, 4, 1, 2,
                                                             3).contiguous()
                B, C, D, H, W = out.shape
                if self.output_2d:
                    outs.append(out[:,:,D//2,:,:].view(B,C,H,W))
                else:
                    outs.append(out.view(B,C,D,H,W))

        return tuple(outs)
