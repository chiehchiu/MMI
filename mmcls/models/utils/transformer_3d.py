# Copyright (c) OpenMMLab. All rights reserved.

# This script consists of several convert functions which
# can modify the weights of model in original repo to be
# pre-trained weights.

from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (build_activation_layer, build_conv_layer,
                              build_norm_layer, xavier_init)
from mmcv.runner.base_module import BaseModule

def nlc_to_ncdhw(x, dhw_shape):
    """Convert [N, L, C] shape tensor to [N, C, D, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        dhw_shape (Sequence[int]): The depth, height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, D, H, W] after conversion.
    """
    D, H, W = dhw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == D * H * W, 'The seq_len does not match D, H, W'
    return x.transpose(1, 2).reshape(B, C, D, H, W).contiguous()


def ncdhw_to_nlc(x):
    """Flatten [N, C, D, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, D, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 5
    return x.flatten(2).transpose(1, 2).contiguous()


def pvt_convert(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    use_abs_pos_embed = False
    use_conv_ffn = False
    for k in ckpt.keys():
        if k.startswith('pos_embed'):
            use_abs_pos_embed = True
        if k.find('dwconv') >= 0:
            use_conv_ffn = True
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm.'):
            continue
        if k.startswith('cls_token'):
            continue
        if k.startswith('pos_embed'):
            stage_i = int(k.replace('pos_embed', ''))
            new_k = k.replace(f'pos_embed{stage_i}',
                              f'layers.{stage_i - 1}.1.0.pos_embed')
            if stage_i == 4 and v.size(1) == 50:  # 1 (cls token) + 7 * 7
                new_v = v[:, 1:, :]  # remove cls token
            else:
                new_v = v
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}',
                              f'layers.{stage_i - 1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            layer_i = int(k.split('.')[1])
            new_layer_i = layer_i + use_abs_pos_embed
            new_k = k.replace(f'block{stage_i}.{layer_i}',
                              f'layers.{stage_i - 1}.1.{new_layer_i}')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                if use_conv_ffn:
                    new_k = new_k.replace('fc2.', '4.')
                else:
                    new_k = new_k.replace('fc2.', '3.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
        elif k.startswith('norm'):
            stage_i = int(k[4])
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i - 1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v

    return new_ckpt

def swin_converter(ckpt, pretrained2d=False):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        if not pretrained2d:  # for 3d weights, load them directly
            return x
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        if not pretrained2d:
            return x
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
            elif 'mlp.' in k:
                if 'mlp.fc1.' in k:
                    new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                elif 'mlp.fc2.' in k:
                    new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                else:
                    new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                elif 'norm.' in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt

class AdaptivePadding3D(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (tuple): Size of the kernel:
        stride (tuple): Stride of the filter. Default: 1:
        dilation (tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    """

    def __init__(self, kernel_size=(1,1,1), stride=(1,1,1), dilation=(1,1,1), padding='corner'):

        super(AdaptivePadding3D, self).__init__()

        assert padding in ('same', 'corner')
        assert len(kernel_size) == 3
        assert len(stride) == 3
        assert len(dilation) == 3

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_d, input_h, input_w = input_shape
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.stride
        output_d = math.ceil(input_d / stride_d)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_d = max((output_d - 1) * stride_d +
                    (kernel_d - 1) * self.dilation[0] + 1 - input_d, 0)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[1] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[2] + 1 - input_w, 0)
        return pad_d, pad_h, pad_w

    def forward(self, x):
        pad_d, pad_h, pad_w = self.get_pad_shape(x.size()[-3:])
        if pad_d>0 or pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h, 0, pad_d])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2,
                    pad_d // 2, pad_d - pad_d // 2
                ])
        return x

class PatchEmbed3D(BaseModule):
    """3D Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed3D.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d.
        kernel_size (tuple): The kernel_size of embedding conv. Default: 1,4,4.
        stride (tuple): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (tuple): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=96,
        conv_type='Conv3d',
        kernel_size=(1,4,4),
        stride=(1,4,4),
        padding='corner',
        dilation=(1,1,1),
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super(PatchEmbed3D, self).__init__(init_cfg=init_cfg)
        assert len(kernel_size) == 3
        assert len(stride) == 3
        assert len(dilation) == 3
        assert padding in ('corner', 'same')

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        self.adap_padding = AdaptivePadding3D(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding)
        # disable the padding of conv
        padding = (0, 0, 0)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            assert len(input_size) == 3
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_d, pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_d, input_h, input_w = input_size
                input_d = input_d + pad_d
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_d, input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

            d_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            h_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            w_out = (input_size[2] + 2 * padding[2] - dilation[2] *
                     (kernel_size[2] - 1) - 1) // stride[2] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, D, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_d * out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_d, out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3], x.shape[4])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size

class PatchMerging3D(BaseModule):
    """ Patch Merging Layer
        nn.Unfold does not support 5D input. Use original impl for 3D transformer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=None,
                 padding='corner',
                 dilation=(1,1,1),
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(stride, int):
            stride = (1, stride, stride)
        assert stride in [(1,2,2), (2,2,2)], "Only support (1,2,2) or (2,2,2) for patch merging"
        assert len(dilation) == 3
        self.stride = stride

        self.sample_dim = stride[0] * stride[1] * stride[2] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, self.sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(self.sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, L, C = x.shape
        D, H, W = input_size
        assert L == D * H * W, 'input feature has wrong size'
        x = x.view(B, D, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if self.stride[0] == 2:
            pad_input = (D % 2 == 1) or pad_input
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))

        B, D, H, W, C = x.shape
        output_size = (D//self.stride[0], H//self.stride[1], W//self.stride[2])

        if self.stride == (1,2,2):
            x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
            x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
            x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
            x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C
        else:
            x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 1::2, 0::2, :]
            x5 = x[:, 0::2, 1::2, 1::2, :]
            x6 = x[:, 1::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C

        x = x.view(B, -1, self.sample_dim)
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size
