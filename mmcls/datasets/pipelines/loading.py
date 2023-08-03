import os.path as osp

import mmcv
import numpy as np
import cv2
import sys
import pdb

from ..builder import PIPELINES


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

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 new_shape=None,
                 file_client_args=dict(backend='disk'), 
                 saved_key='img'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.new_shape = new_shape
        self.saved_key = saved_key

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        # hard code deepwise
        if self.saved_key == 'blood':
            filename = filename.replace('_resized36', '_blood_resized36')

        if self.new_shape is  None:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        else:
            img = cv2.imread(filename, 0)
            img = img.reshape(self.new_shape)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        # modify deepwise
        results[self.saved_key] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadTensorFromFile(object):
    """Load an 3d tensor from file.
        (noted by lzh) : LIDC-IDRI is supported for the time being.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 data_keys='voxel',
                 transpose=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.data_keys = data_keys
        self.transpose = transpose
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        try:
            with np.load(open(filename, 'rb')) as npz:
                #img = npz['voxel']
                img = npz[self.data_keys]
        except:
            print(filename)
            import pdb
            pdb.set_trace()
            print(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        img_size = img.shape
        if self.transpose:
            # Transpose (d, h, w) to (h, w, d)
            if len(img_size) == 3:
                img = np.transpose(img, (1, 2, 0))
                num_channels = 1
            # Transpose (c, d, h, w) to (h, w, d, c)
            elif len(img_size) == 4:
                img = np.transpose(img,(2, 3, 1, 0))
                num_channels = img_size[0]
            else:
                raise NotImplementedError('not implement img size')
        else:
            num_channels = 1 if len(img.shape) <= 3 else img.shape[0]
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadTensorFromFileWithMask(object):
    """Load an 3d tensor from file.
        (noted by lzh) : LIDC-IDRI is supported for the time being.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 data_keys='voxel',
                 transpose=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.data_keys = data_keys
        self.transpose = transpose
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        try:
            with np.load(open(filename, 'rb')) as npz:
                #img = npz['voxel']
                img = npz[self.data_keys]
        except:
            print(filename)
        mask_name = osp.join(results['mask_prefix'],
                            results['img_info']['filename'])
        try:
            mask = np.load(mask_name,mmap_mode='r')
        except:
            print(mask_name)
        if self.to_float32:
            img = img.astype(np.float32)
            mask = mask.astype(np.float32)
        # Transpose (d, h, w) to (h, w, d)
        if self.transpose:
            img = np.transpose(img, (1, 2, 0))
            mask = np.transpose(mask, (1, 2, 0))
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['mask'] = mask
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        #num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        num_channels = 1
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadTensorFromPNG(object):
    """Load an 3d tensor from file.
        (noted by lzh) : LIDC-IDRI is supported for the time being.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 transpose=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.transpose = transpose
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        filename = results['img_info']['filename']
        img = cv2.imread(filename, -1)
        # im.shape =  (dh, w)
        im_shape = img.shape[1]
        img = img.reshape((-1, im_shape, im_shape))
        # Transpose (d, h, w) to (h, w, d)
        if self.transpose:
            img = np.transpose(img, (1, 2, 0))

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        #num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        num_channels = 1
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'file_client_args={self.file_client_args})')
        return repr_str

