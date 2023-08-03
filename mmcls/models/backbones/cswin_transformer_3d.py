# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from mmcv.runner import BaseModule, ModuleList
import numpy as np
import time

from ...mmcv_custom import load_checkpoint
from mmcls.utils import get_root_logger
from ..builder import BACKBONES

import torch.utils.checkpoint as checkpoint


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., init_cfg=None):
        super().__init__(init_cfg)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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


class LePEAttention(BaseModule):
    """
    Extend to 3D.
    Note: resolution, split_size are all in 'dhw' format.
    """
    def __init__(self, dim, resolution=(224,224,224), idx=1, split_size=(7, 7 ,7), dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., fix_grid_size=False, init_cfg=None):
        """Not supported now, since we have cls_tokens now.....
        resolution only work when set fix_grid_size to True.
        """
        super().__init__(init_cfg)
        self.dim = dim
        self.dim_out = dim_out or dim
        self.fix_grid_size = fix_grid_size # Whether fix grid size for different HW
        if isinstance(resolution, int) or isinstance(resolution, float):
            resolution = (resolution,) * 3
        if isinstance(split_size, int) or isinstance(split_size, float):
            split_size = (split_size,) * 3
        assert len(resolution) == 3 and len(split_size) == 3

        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx
        if idx == -1:
            D_sp, H_sp, W_sp  = self.resolution
        elif idx == 0: # horizontal
            D_sp, H_sp, W_sp = self.split_size[0], self.split_size[1], self.resolution[2]
        elif idx == 1: # vertical
            D_sp, H_sp, W_sp = self.split_size[0], self.resolution[1], self.split_size[2]
        elif idx == 2: # 3d
            D_sp, H_sp, W_sp = self.resolution[0], self.split_size[1], self.split_size[2]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        # Note that self.H_sp etc is changed on forward according to feature map size
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.D_sp = D_sp

        stride = 1
        self.get_v = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin3d(self, x):
        B, C, D, H, W = x.shape
        x = img2windows3d(x, self.D_sp, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.D_sp*self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_rpe_3d(self, x, func):
        B, C, D, H, W = x.shape
        D_sp, H_sp, W_sp = self.D_sp, self.H_sp, self.W_sp
        x = x.view(B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().reshape(-1, C, D_sp, H_sp, W_sp) ### B', C, D', H', W'

        rpe = func(x) ### B', C, D', H', W'
        rpe = rpe.reshape(-1, self.num_heads, C // self.num_heads, D_sp * H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        # B' heads N C'
        x = x.reshape(-1, self.num_heads, C // self.num_heads, D_sp * H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, rpe

    def forward(self, temp):
        """
        x: B N C
        mask: B N N
        """
        B, _, C, D, H, W = temp.shape # temp is (B, 3, C, D, H, W), 3 for q,k,v


        idx = self.idx
        
        if not self.fix_grid_size:
            if idx == -1:
                D_sp, H_sp, W_sp = D, H, W
            elif idx == 0:
                D_sp, H_sp, W_sp = self.split_size[0], H, self.split_size[2]
            elif idx == 1:
                D_sp, H_sp, W_sp = self.split_size[0], self.split_size[1], W
            elif idx == 2:
                D_sp, H_sp, W_sp = D, self.split_size[1], self.split_size[2]
            else:
                print ("ERROR MODE in forward", idx)
                exit(0)
            self.D_sp = D_sp
            self.H_sp = H_sp
            self.W_sp = W_sp

        ### padding for split window
        D_pad = (self.D_sp - D % self.D_sp) % self.D_sp
        H_pad = (self.H_sp - H % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - W % self.W_sp) % self.W_sp
        top_pad = H_pad//2
        down_pad = H_pad - top_pad
        left_pad = W_pad//2
        right_pad = W_pad - left_pad
        forward_pad = D_pad//2
        backward_pad = D_pad - forward_pad
        H_ = H + H_pad
        W_ = W + W_pad
        D_ = D + D_pad

        qkv = F.pad(temp, (left_pad, right_pad, top_pad, down_pad, forward_pad, backward_pad)) ### B,3,C,D',H',W'
        qkv = qkv.permute(1, 0, 2, 3, 4, 5)
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        q = self.im2cswin3d(q)
        k = self.im2cswin3d(k)
        v, rpe = self.get_rpe_3d(v, self.get_v)

        ### Local attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)

        attn = self.attn_drop(attn)

        x = (attn @ v) + rpe
        x = x.transpose(1, 2).reshape(-1, self.D_sp*self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img3d(x, self.D_sp, self.H_sp, self.W_sp, D_, H_, W_) # B D_ H_ W_ C
        x = x[:, forward_pad:D+forward_pad, top_pad:H+top_pad, left_pad:W+left_pad, :]
        x = x.reshape(B, -1, C)

        return x

class CSWinBlock(BaseModule):

    def __init__(self, dim, patches_resolution, num_heads,
                 split_size=(7, 7, 7), mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, fix_grid_size=False, init_cfg=None):
        super().__init__(init_cfg)
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = patches_resolution
        if isinstance(split_size, int) or isinstance(split_size, float):
            split_size = (split_size,) * 3
        assert len(split_size) == 3
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.norm1 = norm_layer(dim)

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 3
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        if last_stage:
            self.attns = ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, fix_grid_size=fix_grid_size, init_cfg=None)
                for i in range(self.branch_num)])
        else:
            self.attns = ModuleList([
                LePEAttention(
                    dim//3, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//3, dim_out=dim//3,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, fix_grid_size=fix_grid_size, init_cfg=None)
                for i in range(self.branch_num)])
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop, init_cfg=None)
        self.norm2 = norm_layer(dim)

        atten_mask_matrix = None

        self.register_buffer("atten_mask_matrix", atten_mask_matrix)
        self.H = None
        self.W = None
        self.D = None

    def forward(self, x, dhw_shape=None):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        D = self.D
        H = self.H
        W = self.W
        if D is None:
            D, H, W = dhw_shape  # for mix-arch in swin_lepe
        assert L == D * H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        temp = self.qkv(img).reshape(B, D, H, W, 3, C).permute(0, 4, 5, 1, 2, 3)
        
        if self.branch_num == 3:
            x1 = self.attns[0](temp[:,:,:C//3,:,:,:])
            x2 = self.attns[1](temp[:,:,C//3:2*C//3,:,:,:])
            x3 = self.attns[2](temp[:,:,2*C//3:,:,:,:])
            attened_x = torch.cat([x1,x2,x3], dim=2)
        else:
            attened_x = self.attns[0](temp)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def img2windows3d(img, D_sp, H_sp, W_sp):
    """
    img: B C D H W
    """
    B, C, D, H, W = img.shape
    img_reshape = img.view(B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, D_sp * H_sp * W_sp, C) # B * N * C
    return img_perm

def windows2img3d(img_splits_hw, D_sp, H_sp, W_sp, D, H, W):
    """
    img_splits_hw: B' D H W C
    """
    B = int(img_splits_hw.shape[0] / (D * H * W / D_sp / H_sp / W_sp))

    img = img_splits_hw.view(B, D // D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, D, H, W, -1)
    return img


class Merge_Block(BaseModule):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm, depth_downsamle=True, init_cfg=None):
        super().__init__(init_cfg)
        if depth_downsamle:
            self.conv = nn.Conv3d(dim, dim_out, 3, 2, 1)
        else:
            self.conv = nn.Conv3d(dim, dim_out, 3, (1,2,2), 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x, D, H, W):
        B, new_DHW, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)
        x = self.conv(x)
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x, D, H, W

@BACKBONES.register_module()
class CSWin3D(BaseModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224, 224), stem_depth_downsample=True, in_chans=1, embed_dim=60, depth=[1,2,21,1], split_size = [1,2,7,7], out_indices=(3,),
                 depth_downsamle= [True, True, True], num_heads=[1,2,4,8], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 stem_stride=4,
                 drop_path_rate=0., fix_grid_size=False, hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False, output_2d=False, init_cfg=None):
        self.init_cfg = init_cfg
        super().__init__(init_cfg)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_2d = output_2d
        self.out_indices = out_indices

        heads=num_heads
        self.use_chk = use_chk
        assert isinstance(split_size, list)
        for idx, split_size_ in enumerate(split_size):
            if isinstance(split_size_, int) or isinstance(split_size_, float):
                split_size_ = (split_size_,) * 3
                split_size[idx] = split_size_
            assert len(split_size_) == 3

        assert len(depth_downsamle)==3, "totally three merge block"

        if stem_depth_downsample:
            self.stage1_conv_embed = nn.Sequential(
                nn.Conv3d(in_chans, embed_dim, 7, stem_stride, 3),   # padding should be 3?
                Rearrange('b c d h w -> b (d h w) c', h = img_size[1]//stem_stride, w = img_size[2]//stem_stride),
                nn.LayerNorm(embed_dim)
            )
        else:
            self.stage1_conv_embed = nn.Sequential(
                nn.Conv3d(in_chans, embed_dim, 7, (1,stem_stride, stem_stride), 3), # padding should be 3?
                Rearrange('b c d h w -> b (d h w) c', h = img_size[1]//stem_stride, w = img_size[2]//stem_stride),
                nn.LayerNorm(embed_dim)
            )

        self.norm1 = nn.LayerNorm(embed_dim) if 0 in self.out_indices else nn.Identity()

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        depth_downsample_ratio = pow(stem_stride, stem_depth_downsample)
        self.stage1 = ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], patches_resolution=np.array(img_size)//np.array((depth_downsample_ratio,stem_stride,stem_stride)), mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                fix_grid_size=fix_grid_size, norm_layer=norm_layer, init_cfg=None)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*(heads[1]//heads[0]), depth_downsamle=depth_downsamle[0], init_cfg=None)
        curr_dim = curr_dim*(heads[1]//heads[0])
        self.norm2 = nn.LayerNorm(curr_dim) if 1 in self.out_indices else nn.Identity()

        depth_downsample_ratio = pow(stem_stride, stem_depth_downsample)*pow(2, np.sum(depth_downsamle[:1]))
        self.stage2 = ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], patches_resolution=np.array(img_size)//np.array((depth_downsample_ratio,2*stem_stride,2*stem_stride)), mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:1])+i], 
                fix_grid_size=fix_grid_size, norm_layer=norm_layer, init_cfg=None)
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*(heads[2]//heads[1]), depth_downsamle=depth_downsamle[1], init_cfg=None)
        curr_dim = curr_dim*(heads[2]//heads[1])
        self.norm3 = nn.LayerNorm(curr_dim) if 2 in self.out_indices else nn.Identity()
        temp_stage3 = []
        depth_downsample_ratio = pow(stem_stride, stem_depth_downsample)*pow(2, np.sum(depth_downsamle[:2]))
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], patches_resolution=np.array(img_size)//np.array((depth_downsample_ratio,4*stem_stride,4*stem_stride)), mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:2])+i], 
                fix_grid_size=fix_grid_size, norm_layer=norm_layer, init_cfg=None)
            for i in range(depth[2])])

        self.stage3 = ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*(heads[3]//heads[2]), depth_downsamle=depth_downsamle[2], init_cfg=None)
        curr_dim = curr_dim*(heads[3]//heads[2])
        depth_downsample_ratio = pow(stem_stride, stem_depth_downsample)*pow(2, np.sum(depth_downsamle[:3]))
        self.stage4 = ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], patches_resolution=np.array(img_size)//np.array((depth_downsample_ratio,8*stem_stride,8*stem_stride)), mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[np.sum(depth[:-1])+i], 
                fix_grid_size=fix_grid_size, norm_layer=norm_layer, last_stage=True, init_cfg=None)
            for i in range(depth[-1])])
       
        self.norm4 = norm_layer(curr_dim) if 3 in self.out_indices else nn.Identity()


    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            """
            elif isinstance(m, nn.Conv3d):
                print("=="*80)
                print('Here use Conv3d,initialize')
                print("=="*80)
                nn.init.xavier_uniform(m.weight)
            """

        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, self.init_cfg.checkpoint, strict=False, logger=logger)
        elif self.init_cfg is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def save_out(self, x, norm, D, H, W, get_center=True):
        x = norm(x)
        B, N, C = x.shape
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        if get_center:
            x = x[:,:,D//2,:,:].view(B, C, H, W)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stage1_conv_embed[0](x) ### B, C, D, H, W
        B, C, D, H, W = x.size()
        x = x.reshape(B, C, -1).transpose(-1,-2).contiguous()
        x = self.stage1_conv_embed[2](x)

        out = []
        for blk in self.stage1:
            blk.D = D
            blk.H = H
            blk.W = W
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        out.append(self.save_out(x, self.norm1, D, H, W, get_center=self.output_2d))

        out_indices = 1
        for pre, blocks, norm in zip([self.merge1, self.merge2, self.merge3], 
                                     [self.stage2, self.stage3, self.stage4],
                                     [self.norm2 , self.norm3 , self.norm4 ]):
            x, D, H, W = pre(x, D, H, W)
            for blk in blocks:
                blk.D = D
                blk.H = H
                blk.W = W
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            if out_indices in self.out_indices:
                out.append(self.save_out(x, norm, D, H, W, get_center=self.output_2d))
            out_indices += 1

        return tuple(out)

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


