import inspect
import math
import random

import mmcv
import numpy as np

from ..builder import PIPELINES

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from .io4med import *
import sys
import pdb
import cv2
import shutil

import pdb
import sys

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

# LJ
@PIPELINES.register_module()
class TensorResize(object):
    """Resize images.

    This transform resizes the input image to some scale. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 2 multiscale modes:
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range'):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)


        # given multiple scales or a range of scales
        assert multiscale_mode in ['value', 'range']
        self.multiscale_mode = multiscale_mode

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        edges = [s[0] for s in img_scales]
        scale_edge = np.random.randint(
            min(edges),
            max(edges) + 1)
        img_scale = (scale_edge, scale_edge)
        return img_scale, None


    def _random_scale(self, results):
        if len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):

        img, w_scale, h_scale = self._imresize(
            results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale], dtype=np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['scale_factor'] = scale_factor

    def _imresize(self, img, size, return_scale=False):
        h, w, depth = img.shape[:]
        resized_img = []
        for d in range(depth):
            _img = cv2.resize(
                img[:, :, d], size, interpolation=cv2.INTER_LINEAR)

            resized_img.append(_img)

        resized_img = np.dstack(resized_img)

        if not return_scale:
            return resized_img
        else:
            w_scale = size[0] / w
            h_scale = size[1] / h
            return resized_img, w_scale, h_scale

    def __call__(self, results):

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={})'
                        ).format(self.img_scale, self.multiscale_mode)

        return repr_str


@PIPELINES.register_module()
class TensorFlip(object):
    """Flip the image randomly.

        Flip the image randomly based on flip probaility and flip direction.

        Args:
            flip_prob (float): probability of the image being flipped. Default: 0.5
            direction (str, optional): The flipping direction. Options are
                'x', 'y' and 'z'.
        """

    def __init__(self, flip_probs=[0.5, 0.5, 0.5], directions=['y', 'x', 'z']):
        assert len(flip_probs) == len(directions) == 3
        for flip_prob in flip_probs: assert 0 <= flip_prob <= 1
        # assert len(directions) > 0
        self.flip_probs = flip_probs
        self.directions = directions

    def _flip(self, array, axis):
        '''Flip tensor.
           axis: -1: no flip, 0: Y-axis, 1: X-axis, 2: Z-axis
        '''
        ref = np.flip(array, axis)

        return ref

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results
        """

        voxel = results['img']

        flips = []
        for dir in self.directions:
            idx = self.directions.index(dir)
            flip_prob = self.flip_probs[idx]
            flip = True if np.random.rand() < flip_prob else False
            flips.append(flip)
            # flip image
            if flip:
                voxel = self._flip(voxel, axis=idx)

        results['img'] = voxel

        if True in flips:
            results['flips'] = flips

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(Flip_Transform)'
        return repr_str


@PIPELINES.register_module()
class TensorZRotation(object):
    """rotate the image randomly.

    rotate the image randomly based on rotation probaility.

    only rotate in Z'-axis
    """

    def __init__(self, rotation_prob=0.5):
        assert 0 <= rotation_prob <= 1
        self.rotation_prob = rotation_prob

    def _rotate(self, array, angle):
        '''rotate tensor.
           array h, w, d
           axis: (0, 1) rotate in Z'-axis
        '''

        ref = np.rot90(array, angle, (0, 1))

        return ref

    def __call__(self, results):
        """Call function to rotate image.

         Args:
             results (dict): Result dict from loading pipeline.

         Returns:
             dict: rotated results
         """
        voxel = results['img']

        rotate = True if np.random.rand() < self.rotation_prob else False
        # rotate image
        if rotate:
            # angle = np.random.randint(4)
            angle = 2
            voxel = self._rotate(voxel, angle)

        results['img'] = voxel

        if rotate:
            results['rotate'] = rotate
            results['angle'] = angle

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(Rotation_Transform)'
        return repr_str


@PIPELINES.register_module()
class TensorCrop:
    def __init__(self, crop_size, move=5, train=True):
        self.size = (crop_size, ) * 3 if isinstance(crop_size, int) else crop_size
        self.move = move
        self.train = train

    def crop_xy(self, array, center, size):
        # For input tensor (h, w, d), crop along xy axis

        y, x, z = center
        h, w, d = size
        cropped = array[y - h // 2:y + h // 2,
                  x - w // 2:x + w // 2,
                  ...]
        # cropped = array[y - h // 2:y + h // 2,
        #           x - w // 2:x + w // 2,
        #           z - d // 2: z + d // 2]

        return cropped

    def random_center(self, shape, move):
        offset = np.random.randint(-move, move + 1, size=3)
        yxz = np.array(shape) // 2 + offset
        return yxz

    def __call__(self, results):
        voxel = results['img']

        shape = voxel.shape
        # norm
        if self.train:
            if self.move is not None:
                center = self.random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = self.crop_xy(voxel, center, self.size)
        else:
            center = np.array(shape) // 2
            voxel_ret = self.crop_xy(voxel, center, self.size)

        results['img'] = voxel_ret

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(Crop_Transform)'
        return repr_str


@PIPELINES.register_module()
class TensorXYCrop:
    def __init__(self, crop_size, move=5, train=True):
        self.size = (crop_size, ) * 2 if isinstance(crop_size, int) else crop_size
        self.move = move
        self.train = train

    def crop_xy(self, array, center, size):
        # For input tensor (h, w, d), crop along xy axis

        y, x, z = center
        h, w = size
        cropped = array[y - h // 2:y + h // 2,
                  x - w // 2:x + w // 2,
                  ...]

        return cropped

    def random_center(self, shape, move):
        offset = np.random.randint(-move, move + 1, size=3)
        yxz = np.array(shape) // 2 + offset
        return yxz

    def get_crop_center(self, voxel):
        shape = voxel.shape
        # norm
        if self.train:
            if self.move is not None:
                center = self.random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
        else:
            center = np.array(shape) // 2

        return center

    def _crop_img(self, results, center):
        results['img'] = self.crop_xy(results['img'], center, self.size)

    def _crop_seg(self, results, center):
        results['gt_semantic_seg'] = self.crop_xy(results['gt_semantic_seg'], center, self.size)

    def __call__(self, results):

        center = self.get_crop_center(results['img'])
        self._crop_img(results, center)

        if 'gt_semantic_seg' in results:
            self._crop_seg(results, center)

            if 'gt_label' in results:
                # crop后判断是否更新gt_lable np.sum()费时?
                _label = 0 if np.sum(results['gt_semantic_seg']) == 0 else 1
                results['gt_label'] = np.array(_label, dtype=np.int64)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(Crop_Transform)'
        return repr_str


@PIPELINES.register_module()
class TensorMultiZCrop:
    def __init__(self, num_slice, stride=0):
        self.stride = stride
        self.num_slice = num_slice

    def crop_z(self, array, z_start, z_end):
        # For input tensor (h, w, d), crop along xy axis

        cropped = array[..., z_start:z_end]

        return cropped

    def get_params(self, voxel):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        h, w, depth = voxel.shape[:]
        assert depth >= self.num_slice

        params = []
        z_start, z_end = 0, self.num_slice
        while z_end < depth:
            params.append([z_start, z_end])
            z_start = z_start + self.stride
            z_end = z_start + self.num_slice
        # 尾部对齐
        params.append([depth - self.num_slice, depth])

        return params

    def __call__(self, results):
        voxel = results['img']

        voxel_ret = []
        params = self.get_params(voxel)
        assert isinstance(params, list)
        for i in range(len(params)):
            z_start, z_end = params[i]
            voxel_ = self.crop_z(voxel, z_start, z_end)
            voxel_ret.append(voxel_)

        voxel_ret = np.stack(voxel_ret)

        results['img'] = voxel_ret

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(ZCrop_Transform)'
        return repr_str


@PIPELINES.register_module()
class TensorRandomCrop:
    def __init__(self, crop_size, extra_move=1, train=True):
        self.size = (crop_size, ) * 3 if isinstance(crop_size, int) else crop_size
        self.train = train
        self.extra_move = extra_move

    def crop_xy(self, array, x_start, y_start, x_end, y_end):
        # For input tensor (h, w, d), crop along xy axis

        cropped = array[y_start : y_end,
                  x_start : x_end,
                  ...]

        return cropped

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size[:2]
        if width == target_width and height == target_height:
            x_start, y_start, x_end, y_end = 0, 0, height, width
            return x_start, y_start, x_end, y_end

        x_start = random.randint(0, height - target_height)
        y_start = random.randint(0, width - target_width)

        x_end = x_start + target_width
        y_end = y_start + target_height

        return x_start, y_start, x_end, y_end

    def get_params_test(self, img, output_size, extra_move=1):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size[:2]
        if width == target_width and height == target_height:
            x_start, y_start, x_end, y_end = 0, 0, height, width
            return [x_start, y_start, x_end, y_end ]

        # 向下取整
        move_y = height // target_height + extra_move
        stride_y = (height - target_height) // (move_y - 1)

        move_x = width // target_width + extra_move
        stride_x = (width - target_width) // (move_x - 1)

        params = []
        y_start = -stride_y
        for i in range(move_y):
            y_start = y_start + stride_y
            x_start = -stride_x
            for j in range(move_x):
                x_start = x_start + stride_x

                x_end = x_start + target_width
                y_end = y_start + target_height

                params.append([x_start, y_start, x_end, y_end])

        return params

    def __call__(self, results):
        voxel = results['img']

        if self.train:
            x_start, y_start, x_end, y_end = self.get_params(voxel, self.size)
            voxel_ret = self.crop_xy(voxel, x_start, y_start, x_end, y_end)
        else:
            voxel_ret = []
            params = self.get_params_test(voxel, self.size, extra_move=self.extra_move)
            assert isinstance(params, list)
            for i in range(len(params)):
                x_start, y_start, x_end, y_end = params[i]
                voxel_ = self.crop_xy(voxel, x_start, y_start, x_end, y_end)
                voxel_ret.append(voxel_)

            voxel_ret = np.stack(voxel_ret)

        results['img'] = voxel_ret

        # vis(results, tag='randomcrop')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(RandomCrop_Transform)'
        return repr_str



@PIPELINES.register_module()
class TensorNorm:
    def __init__(self, mean=99., std=77.):
        self.mean = mean
        self.std = std

    def __call__(self, results):

        voxel = results['img']

        voxel = (voxel - self.mean) / self.std
        if len(voxel.shape) < 4:
            voxel = np.expand_dims(voxel, 0).astype(np.float32)

        results['img'] = voxel

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(Norm_Transform)'
        return repr_str
