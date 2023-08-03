import os
import os.path as osp
import numpy as np
#import nibabel as nb
from PIL import Image
from pathlib import Path
import time

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())


def convert_label(label, label_mapping=None, inverse=False):
    if label_mapping is None:
        return label
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


def open_mask_random(fp):
    """
    more than one mask may exist for a image, as 160_{0,1,9}.png
    during training, randomly load one
    """
    fp = str(fp)
    pil_load = lambda fn: np.array(Image.open(fn))
    # fp = Path(fp)
    fn = fp.split('/')[-1].split('_')[0]
    fn_dir = '/'.join(fp.split('/')[:-1])
    fp_by_doc = lambda i: '/'.join([fn_dir, str('%s_%d.png' % (fn, i))])

    if osp.exists(fp_by_doc(9)):
        return pil_load(fp)
    else:
        fp_candidates = [fp_by_doc(i) for i in (0, 1)]
        if np.random.random() > 0.5: fp_candidates = fp_candidates[::-1]
        for cur_fp in fp_candidates:
            if osp.exists(cur_fp):
                return pil_load(cur_fp)
    return pil_load(fp)


pil_load = lambda fn: np.array(Image.open(str(fn)))
get_s_ix = lambda fn: int(fn.stem.split('/')[0])


class LoadMultipleSlices(object):
    """
    given the file path of a slice and number of slices,
    load a volume centered at the given slice
    The slices will be stacked in the last dimension to accommondate later transformations


    """

    def __init__(self, fp, nb_channels=3, skip_step=1,
                 is_duplicates=False, verbose=False, pid_info=None) -> None:
        self.fp = Path(fp)
        self.file_dir = self.fp.parent
        self.fn = self.fp.name
        self.nb_channels = nb_channels
        self.skip_step = skip_step
        self.is_duplicates = is_duplicates
        self.verbose = verbose
        # print(self.file_dir)
        self.all_fns_dict = None if pid_info is None else pid_info[str(self.file_dir)]

        self.is_mask = '_' in self.fp.stem
        self.image_class = int(self.fp.stem.split('_')[1]) if self.is_mask else None

        # if self.verbose: print('fn %s, is_mask %s, class %s' %(self.fn, self.is_mask, self.image_class))

    def get_fps2load(self):

        if self.nb_channels == 1:
            return [self.fp], None
        pre_nb = int((self.nb_channels - 0.001) // 2)
        post_nb = self.nb_channels - pre_nb - 1
        tg_ix = int(self.fp.stem[:3])

        if self.all_fns_dict is None:
            all_fns = os.listdir(self.file_dir)
            all_fns_dict = {int(str(f)[:3]): f for f in all_fns}
        else:
            all_fns_dict = self.all_fns_dict
            assert len(all_fns_dict) > 0
        nb_slices = max(list(all_fns_dict)) + 1  # slice index start from 0
        # tg_ix = 398
        pre_ixs = [max(tg_ix - (pre_nb - i) * self.skip_step, 0) for i in range(pre_nb)]
        post_ixs = [min(tg_ix + (i + 1) * self.skip_step, nb_slices - 1) for i in range(post_nb)]
        ixs = pre_ixs + [tg_ix] + post_ixs

        if self.verbose: print(self.fn, ixs)

        fps = [self.file_dir / all_fns_dict[i] if i in all_fns_dict else self.fp for i in ixs]

        ocurrance = lambda x: sum([1 for a in ixs if a == x]) - 1
        z_bound_count = -1 * max(0, ocurrance(0)) + 1 * max(0, ocurrance(nb_slices - 1))
        if self.verbose: print('\ttg_ix:%d\ttotal%d\tipch:%d\tzbound:%d' % (
            tg_ix, nb_slices, self.nb_channels, z_bound_count))
        return fps, z_bound_count

    def _load_images(self, fps, load_func):
        center_fp = fps[len(fps) // 2]
        x_list = []
        for fp in fps:
            fn_i = center_fp if not osp.exists(fp) else fp
            # if self.input_duplicates:
            x = load_func(fn_i)
            x_list.append(x)
            # print('\n')
            # print('tg %s actual %s' %(tg_ix, actual_ixs))
        x = np.stack(x_list, axis=-1)
        return x

    def load(self, is_z_first=False, use_med_view=False):
        fps, z_bound_count = self.get_fps2load()
        load_func = open_mask_random if self.is_mask else pil_load
        image = self._load_images(fps, load_func)

        if use_med_view:
            fps_med = [str(fp).replace('image_links', 'image_links_med') for fp in fps]
            assert all([osp.exists(f) for f in fps_med]), \
                'Medview file not exists %s' % str(fps_med[self.nb_channels // 2])
            image_med = self._load_images(fps_med, load_func)
            image = np.concatenate([image, image_med], axis=-1)
        if is_z_first:
            image = np.moveaxis(image, -1, 0)
        return image, z_bound_count


def mkdir(path):
    # credit goes to YY
    # 判别路径不为空
    path = str(path)
    if path != '':
        # 去除首空格
        path = path.rstrip(' \t\r\n\0')
        if '~' in path:
            path = os.path.expanduser(path)
        # 判别是否存在路径,如果不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)


def array2nii(mask_3d, store_root, file_name, affine_matrix=None, nii_obj=None,
              is_compress=True):
    """
    将输入图像存储为nii
    输入维度是z, r=y, c=x
    输出维度是x=c, y=r, z
    :param mask_3d:
    :param store_root:
    :param file_name:
    :param nii_obj:
    :return:
    """

    extension = 'nii.gz' if is_compress else 'nii'
    mask_3d = np.swapaxes(mask_3d, 0, 2)
    # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
    store_path = osp.join(store_root, '.'.join([file_name, extension]))
    if nii_obj is None:
        if affine_matrix is None: affine_matrix = np.eye(4)
        # print(affine_matrix)
        nb_ojb = nb.Nifti1Image(mask_3d, affine_matrix)
    else:
        nb_ojb = nb.Nifti1Image(mask_3d, nii_obj.affine, nii_obj.header)

    nb.save(nb_ojb, store_path)
    return store_path
    # print(' done saving nii to ', store_path


def load_image_nii(img_nii_fp, verbose=True):
    nii_obj = nb.load(img_nii_fp)
    affine_matrix = nii_obj.affine
    x_col_sign = int(np.sign(-1 * affine_matrix[0, 0]))
    y_row_sign = int(np.sign(-1 * affine_matrix[1, 1]))
    if verbose: print('x y sign', x_col_sign, y_row_sign)
    if verbose: print('affine matrix\n', affine_matrix)
    image_3d = np.swapaxes(nii_obj.get_data(), 2, 0)

    image_3d = image_3d[:, ::y_row_sign, ::x_col_sign]
    # spacing_list = nii_obj.header.get_zooms()[::-1]
    return image_3d, affine_matrix


view2permute = {'saggital': (0, 1, 2),
                'coronal': (2, 0, 1),
                'axial': (1, 2, 0)
                }

# normally z axis should be in first dimension
view2axis = {'saggital': 'xzy',
             'coronal': 'yzx',
             'axial': 'zyx'}
# nii dataset always assume xyz dimension order
axis_order_map = {'xzy': (0, 2, 1),
                  'zyx': (2, 1, 0),
                  'yzx': (1, 2, 0),
                  None: None
                  }

axis_reorder_map = {'xzy': (0, 2, 1),
                    'zyx': (2, 1, 0),
                    'yzx': (2, 0, 1),
                    None: None}

# xyz-> xcyz -> x=0,c=1,y=2,z=3
axis_reorder_map4d = {'xzy': (0, 1, 3, 2),  # xczy xcyz
                      'zyx': (3, 1, 2, 0),  # zcyx xcyz
                      'yzx': (3, 1, 0, 2)}  # yczx xcyz


# image and mask are already saved as nii

class IO4Nii(object):

    @staticmethod
    def read(img_nii_fp, verbose=True, axis_order='zyx'):
        assert axis_order in list(axis_order_map)
        nii_obj = nb.load(img_nii_fp)
        affine_matrix = nii_obj.affine
        x_col_sign = int(np.sign(-1 * affine_matrix[0, 0]))
        y_row_sign = int(np.sign(-1 * affine_matrix[1, 1]))
        if verbose: print('x y sign', x_col_sign, y_row_sign)
        if verbose: print('affine matrix\n', affine_matrix)
        image_3d = nii_obj.get_data()
        image_3d = image_3d[::x_col_sign, ::y_row_sign, :]
        permute_order = axis_order_map[axis_order]
        if permute_order is not None:
            image_3d = image_3d.transpose(permute_order)
        # spacing_list = nii_obj.header.get_zooms()[::-1]
        return image_3d, affine_matrix

    @staticmethod
    def read_ww(img_nii_fp, axis_order='zyx',
                ww=400, wc=50, is_uint8=True, verbose=True):

        image_new, affine_matrix = IO4Nii.read(img_nii_fp,
                                               axis_order=axis_order,
                                               verbose=verbose)
        if isinstance(ww, int) and isinstance(wc, int):
            image_new = adjust_ww_wl(image_new, ww, wc, is_uint8)
        return image_new, affine_matrix

    @staticmethod
    def read_shape_xyz(img_nii_fp, verbose=False):
        nii_obj = nb.load(img_nii_fp)
        image_shape_xyz = nii_obj.header.get_data_shape()
        # image_shape_zyx = image_shape_xyz[::-1]
        if verbose: print(image_shape_xyz)
        return image_shape_xyz

    @staticmethod
    def write(mask_3d, store_root, file_name, affine_matrix=None, nii_obj=None,
              is_compress=True, axis_order='zyx'):
        """
        将输入图像存储为nii
        输入维度是z, r=y, c=x
        输出维度是x=c, y=r, z
        :param mask_3d:
        :param store_root:
        :param file_name:
        :param nii_obj:
        :return:
        """
        extension = 'nii.gz' if is_compress else 'nii'
        permute_order = axis_reorder_map[axis_order]
        if permute_order is not None:
            mask_3d = mask_3d.transpose(permute_order)
        # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
        store_path = osp.join(store_root, '.'.join([file_name, extension]))
        if nii_obj is None:
            if affine_matrix is None: affine_matrix = np.eye(4)
            x_col_sign = int(np.sign(-1 * affine_matrix[0, 0]))
            y_row_sign = int(np.sign(-1 * affine_matrix[1, 1]))
            # print(affine_matrix)
            nb_ojb = nb.Nifti1Image(mask_3d[::x_col_sign, ::y_row_sign, :], affine_matrix)
        else:
            nb_ojb = nb.Nifti1Image(mask_3d, nii_obj.affine, nii_obj.header)

        nb.save(nb_ojb, store_path)
        return store_path


# @staticmethod
def adjust_ww_wl(image, ww=250, wc=250, is_uint8=True):
    """
    调整图像得窗宽窗位
    :param image: 3D图像
    :param ww: 窗宽
    :param wl: 窗位
    :return: 调整窗宽窗位后的图像
    """
    min_hu = wc - (ww / 2)
    max_hu = wc + (ww / 2)
    new_image = np.clip(image, min_hu, max_hu)  # np.copy(image)
    if is_uint8:
        new_image -= min_hu
        new_image = np.array(new_image / ww * 255., dtype=np.uint8)

    return new_image


class ImageDrawerDHW(object):
    """
    start with a image tensor, such as CT volume in D, H, W
    give an index of slice and nb_channels, return a subset of slices centered on that index

    index from the first dimension
    put the indexing dim to the last dim as channels
    """

    def __init__(self, image_tensor, nb_channels=3, dim0_to_last=True, skip_step=1,
                 fp='', verbose=False) -> None:
        self.image = image_tensor
        self.nb_channels = nb_channels
        self.dim0_to_last = dim0_to_last
        self.skip_step = skip_step
        self.fp = fp
        self.verbose = verbose

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        assert isinstance(index, int)
        assert (index >= 0) and (index < len(self)), print_tensor(f'fp"{self.fp} ix: {index}', self.image)
        pre_nb = int((self.nb_channels - 0.001) // 2)
        post_nb = self.nb_channels - pre_nb - 1
        tg_ix = index
        nb_slices = len(self)
        # tg_ix = 398
        pre_ixs = [max(tg_ix - (pre_nb - i) * self.skip_step, 0) for i in range(pre_nb)]
        post_ixs = [min(tg_ix + (i + 1) * self.skip_step, nb_slices - 1) for i in range(post_nb)]
        ixs = pre_ixs + [tg_ix] + post_ixs
        ip_slices = self.image[ixs, ...]
        if self.dim0_to_last: ip_slices = np.moveaxis(ip_slices, 0, -1)
        return ip_slices, ixs
