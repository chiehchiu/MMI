import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import os
import os.path as osp
import pdb
import sys
import itertools
import numpy as np
from ..datasets.pipelines.io4med import *
import torch.nn.functional as F
from mmcls.models.utils.cam import CAM
import cv2
from scipy import ndimage

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

def single_gpu_test(model, data_loader, show=False, out_dir=None, output_cam = None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        if show or out_dir:
            pass  # TODO

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collectio)n. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


# LJ
def single_gpu_test_cam3d(model, data_loader, cam2nii=None, cam2jpg=None, is_2d=False):
    model.eval()

    data_infos = data_loader.dataset.data_infos
    gt_labels = [data_info['gt_label'] for data_info in data_infos]
    img_infos = [data_info['img_info'] for data_info in data_infos]
    print('-'*20)
    backbone_name  = model.module.backbone.__class__.__name__
    if backbone_name == 'ResNet3D':
        if len(model.module.backbone.layer4) == 2:
            target_layer = model.module.backbone.layer4[-1].relu # conv2
        else:
            target_layer = model.module.backbone.layer4[-1].relu # conv3
    elif backbone_name == 'SwinLePETransformer3D':
        target_layer = model.module.backbone.norm3
    else:
        raise NotImplementedError(f"Not Implement cam for {backbone_name}")
    weight_fc = model.module.head.fc.parameters().__next__()
    # weight_fc = weight_fc.to('cpu').data

    cam = CAM(target_layer, weight_fc)
    results = []
    class_names = data_loader.dataset.CLASSES
    NIIwriter = CAMWritterNii(cam2nii)
 
    if not is_2d:
        JPGwriter  = CAMWritterJpg(cam2jpg, class_names)
    else:
        JPGwriter  = CAMWritterJpg2D(cam2jpg, class_names)
    start_idx = 0
    for i, data in enumerate(data_loader):
        batch_n = len(data['img'])
        link_paths = [data_meta['ori_filename'] for data_meta in data['img_metas'].data[0]]

        start_idx = start_idx + batch_n
        data_len = data['img'].shape[0]
        with torch.no_grad():
            result = model(return_loss=False, **data)
            # result, cam5d = model(return_loss=False, output_cam=cam2jpg, **data)

        results.append(result)

        print('start_idx : %d'%start_idx)
        # T H W
        img_tensor = data['img']
        size = tuple(img_tensor[0].shape[-3:])
        if backbone_name == 'SwinLePETransformer3D':
            _, _, c = cam.values.activations.size()
            # convert to B C D H W
            down_ratio_d,down_ratio_h,down_ratio_w = 1, 1, 1
            for stride in model.module.backbone.strides:
                down_ratio_d *= stride[0]
                down_ratio_h *= stride[1]
                down_ratio_w *= stride[2]
            d,h,w = size[0]//down_ratio_d, size[1]//down_ratio_h, size[2]//down_ratio_w
            cam.values.activations = cam.values.activations.view(-1,d,h,w,c).permute(0, 4, 1, 2, 3).contiguous()
        if is_2d:
            cam_tensor = cam.getCAM(size)
        else:
            # Use 3D global normalization(-min/max)
            cam_tensor = cam.getCAM()
        if cam2nii:
            NIIwriter.write(img_tensor, cam_tensor)
        if cam2jpg:
            JPGwriter.write(img_tensor, cam_tensor, gt_labels, \
                    result_list=result, class_names=class_names, size=None,link_path=link_paths)
    return results

class CAMWritterNii(object):
    def __init__(self, save_root) -> None:

        self.case_id = 0
        self.save_root = save_root
        assert isinstance(save_root, (type(None), str))

    def _create_dir(self):
        save_dir = os.path.join(self.save_root, '%04d' % self.case_id)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)
        return save_dir

    def write(self, img, cam):
        # assume that the image and att_map are 5D tensors
        img4d = img.squeeze(1).numpy()
        cam5d = cam.cpu().data.numpy()
        batch, cls_num = cam5d.shape[:2]
        for b in range(batch):
            img_b = img4d[b]  # d,h,w
            save_dir = self._create_dir()
            IO4Nii.write(img_b, save_dir, 'image%dx%dx%d' %(img_b.shape))
            for c in range(cls_num):
                cam_b_c = cam5d[b, c, ...]
                cam_b_c -= np.min(cam_b_c)
                cam_b_c /= np.max(cam_b_c)
                cam_b_c = np.array(cam_b_c, dtype=np.float32)
                IO4Nii.write(cam_b_c, save_dir, 'CAM4cls%04d' % c)
            self.case_id += 1


class CAMWritterJpg(object):
    def __init__(self, save_root, class_names) -> None:

        self.case_id = 0
        self.save_root = save_root
        self.class_names = class_names
        assert isinstance(save_root, (type(None), str))

    def _create_dir(self, cls_idx):
        save_dir = os.path.join(self.save_root, '%04d' % self.case_id)
        #save_dir = os.path.join(save_dir, '%02d' % cls_idx)
        save_dir = os.path.join(save_dir, self.class_names[cls_idx])
        #print(self.save_root, self.case_id, cls_idx)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)
        return save_dir

    #CAMWritterJpg   write 
    def write(self, img, cam, gts, result_list, class_names, link_path,size=None):
        # assume that the image and att_map are 5D tensors
        img4d = img.squeeze(1).numpy() # b,d,h,w
        #cam5d = cam.cpu().data.numpy() # b,c,d,h,w
        cam5d = cam
        batch, cls_num, depth = cam5d.shape[:3]
        depth = img4d.shape[1]
        hblank = np.ones((10, 224, 3)) * 255
        #hblank = np.ones((10, 32, 3)) * 255
        vblank = np.ones((224+10+224+10+224, 10, 3)) * 255
        #if not os.path.exists(self.save_root):
        #    os.makedirs(self.save_root)
        for b in range(batch):
            cur_gt = gts[self.case_id]
            #pos_indices = np.where(gts[self.case_id]==0)[0]
            #pos_indices = np.where(gts[self.case_id]==1)[0]
            #c_num = result_list[b].argmax()
            #pd_tags = [self.class_names[pd_i] if prob >=0.5 else '-' for pd_i, prob in enumerate(result_list[b])]
            pd_probs = result_list[b]
            #for c in pos_indices:#range(10):#pos_indices:
            for c in range(cls_num):
                print('----')
                save_path = os.path.join(self.save_root,link_path[b],class_names[c],\
                        '%08d_' % self.case_id  + 'cls%d_'% c + 'g%d_'%cur_gt[c]+'p%.2f.jpg' % pd_probs[c])
                print(save_path)
                print('----')
                if not osp.exists(osp.join(self.save_root,link_path[b],class_names[c])):
                    os.makedirs(osp.join(self.save_root,link_path[b],class_names[c]))
                img_list = []
                heatmap_list = []
                depth_list = []
                cam3d = cam5d[b,c,...]
                size = tuple(img4d[0].shape[-3:])
                cam3d = (cam3d - cam3d.min()) / (cam3d.max() - cam3d.min())
                cam3d = F.interpolate(cam3d[None, None, ...], size=size, mode="trilinear", align_corners=False)
                cam3d = cam3d.cpu().data.numpy()
                for d in range(depth):
                    depth_list.append(d)
                    img2d = img4d[b, d, ...] # h,w
                    #cam2d = cam5d[b, c, d, ...] # h,w
                    cam2d = cam3d[0, 0, d, ...] # h,w
                    #cam2d -= np.min(cam2d)
                    #cam2d /= np.max(cam2d)

                    cam2d = cam2d * 255.0
                    # if we donot interpolate feature map in function getCAM(), resize it here.
                    cam2d = cam2d.astype(np.uint8)
                    if size:
                        cam2d = cv2.resize(cam2d, size[1:]) # size:(3, 224, 224) --> (224,224)
                    cam2d = ndimage.gaussian_filter(cam2d, sigma=3)
                    heatmap = cv2.applyColorMap(np.uint8(cam2d), cv2.COLORMAP_JET)
                    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    # heatmap = heatmap / 255.0
                    img2d = cv2.cvtColor(img2d, cv2.COLOR_GRAY2BGR)
                    img2d -= np.min(img2d)
                    img2d /= np.max(img2d)
                    img2d = img2d * 255.0

                    vis = heatmap * 0.4 + img2d * 0.6
                    vis = np.concatenate((heatmap, hblank, vis, hblank, img2d),0)
                    img_list.append(vis)
                    heatmap_list.append(heatmap)
                    #vis = np.vstack((heatmap, img2d))
                #print(len(heatmap_list))
                for idx in range(len(heatmap_list)):
                    final_save_path = save_path.replace('.jpg', '_%02d.jpg'%depth_list[idx])
                    cv2.imwrite(final_save_path, img_list[idx])

            self.case_id += 1


class CAMWritterJpg2D(object):
    def __init__(self, save_root, class_names) -> None:

        self.case_id = 0
        self.save_root = save_root
        self.class_names = class_names
        assert isinstance(save_root, (type(None), str))

    def _create_dir(self, cls_idx):
        save_dir = os.path.join(self.save_root, '%08d' % self.case_id)
        save_dir = os.path.join(save_dir, self.class_names[cls_idx])
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)
        return save_dir


    def write(self, img, cam, gts, result_list, class_names, size=None):
        img3d = img.squeeze(1).numpy() # b,h,w
        cam4d = cam.cpu().data.numpy() # b,c,h,w
        batch, cls_num = cam4d.shape[:2]
        hblank = np.ones((10, 224, 3)) * 255
        vblank = np.ones((224+10+224+10+224, 10, 3)) * 255
        for b in range(batch):
            pos_indices = np.where(gts[self.case_id]==1)[0]
            for c in pos_indices:
                 if c not in [3,4,9,10,11,12,13]: #[7]:
                     continue
                 if result_list[b][c] < 0.010: # 0.8
                     #print('case %04d ignore small scored samples for class %s' % (self.case_id, self.class_names[c]))
                     continue
                 save_path = os.path.join(self.save_root, self.class_names[c],\
                        's%.2f_' % result_list[b][c] + '%02d_' % c + \
                        self.class_names[c].replace(' ','_').replace(',','_') +\
                        '_%08d.jpg' % self.case_id )
                 if not osp.exists(osp.join(self.save_root, self.class_names[c])):
                     os.makedirs(osp.join(self.save_root, self.class_names[c]))
                 img2d = img3d[b, ...] # h,w
                 cam2d = cam4d[b, c, ...] # h,w
    
                 cam2d -= np.min(cam2d)
                 cam2d /= np.max(cam2d)
                 cam2d = cam2d * 255.0
                 # if we donot interpolate feature map in function getCAM(), resize it here.
                 cam2d = cam2d.astype(np.uint8)
                 if size:
                     cam2d = cv2.resize(cam2d, size[1:]) # size:(3, 224, 224) --> (224,224)
                 cam2d = ndimage.gaussian_filter(cam2d, sigma=3)
    
                 heatmap = cv2.applyColorMap(np.uint8(cam2d), cv2.COLORMAP_JET)
                 # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                 # heatmap = heatmap / 255.0
    
                 #img2d = cv2.cvtColor(img2d, cv2.COLOR_GRAY2BGR)
                 img2d = img2d.transpose((1,2,0))
                 img2d -= np.min(img2d)
                 img2d /= np.max(img2d)
                 img2d = img2d * 255.0
    
                 vis = heatmap * 0.4 + img2d * 0.6
                 vis = np.concatenate((heatmap, hblank, vis, hblank, img2d),0)
                 print(save_path)    
                 cv2.imwrite(save_path, vis)
    
            self.case_id += 1

