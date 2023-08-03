# Copyright (c) OpenMMLab. All rights reserved.
import torch
import random
import math
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmcls.datasets import SAMPLERS


@SAMPLERS.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == self.num_samples

        return iter(indices)

@SAMPLERS.register_module()
class DistributedSIDSampler(_DistributedSampler):
    """需要注意的是，一个epoch的iter数还是决定于dataset的长度；目前这个在dataset长度的数据被遍历完之前，sampler会被多次调用"""
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        data_infos = dataset.data_infos
        self.sub_dirs = [data_info['img_info']['filename'] for data_info in data_infos]
        self.sids = ['/'.join(sub_dir.split('/')[:2]) for sub_dir in self.sub_dirs]
        self.sids = list(set(self.sids))
        print('length of sid samples are ', len(self.sids), math.ceil(len(self.sids) / self.num_replicas) * self.num_replicas)
        if self.round_up:
            self.total_size = math.ceil(len(self.sids) / self.num_replicas) * self.num_replicas
        else:
            self.total_size = len(self.sids)

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = []
        exist_sid = []
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            total_indices = torch.randperm(len(self.sub_dirs), generator=g).tolist()
        else:
            total_indices = torch.arange(len(self.sub_dirs)).tolist()

        for idx in total_indices:
            sub_dir = self.sub_dirs[idx]
            sid = '/'.join(sub_dir.split('/')[:2])
            if sid not in exist_sid:
                indices.append(idx)
                exist_sid.append(sid)
        #print(len(indices), self.total_size)
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == math.ceil(len(self.sids) / self.num_replicas)

        return iter(indices)

@SAMPLERS.register_module()
class Distributed3DPatchSampler(_DistributedSampler):
    """Used to sample one 2D patch from any certain 3D patch in one epoch"""
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        data_infos = dataset.data_infos
        self.sub_dirs = [data_info['img_info']['filename'] for data_info in data_infos]
        self.sids = [sub_dir.split('/')[0] + '_' + '_'.join(sub_dir.split('/')[1].split('_')[7:10]) for sub_dir in self.sub_dirs]
        self.sids = list(set(self.sids))
        print('length of 3D patches is ', len(self.sids), math.ceil(len(self.sids) / self.num_replicas) * self.num_replicas)
        if self.round_up:
            self.total_size =  math.ceil(len(self.sids) / self.num_replicas) * self.num_replicas
        else:
            self.total_size = len(self.sids)

    def __iter__(self):
        # deterministically shuffle based on epoch
        indices = []
        exist_sid = []
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            total_indices = torch.randperm(len(self.sub_dirs), generator=g).tolist()
        else:
            total_indices = torch.arange(len(self.sub_dirs)).tolist()

        for idx in total_indices:
            sub_dir = self.sub_dirs[idx]
            sid = sub_dir.split('/')[0] + '_' + '_'.join(sub_dir.split('/')[1].split('_')[7:10])
            if sid not in exist_sid:
                indices.append(idx)
                exist_sid.append(sid)
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        #print('sample %d indices in this sampler iter' % len(indices))
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == math.ceil(len(self.sids) / self.num_replicas)

        return iter(indices)

@SAMPLERS.register_module()
class DistributedCTSampler(_DistributedSampler):
    """需要注意的是，一个epoch的iter数还是决定于dataset的长度；目前这个在dataset长度的数据被遍历完之前，sampler会被多次调用"""
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True,
                 num_per_ct=64,
                 pos_fraction=0.5,
                 ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        data_infos = dataset.data_infos
        self.num_per_ct = num_per_ct
        self.pos_fraction = pos_fraction
        self.ct_dict = {}
        for  idx, data_info in enumerate(data_infos):
            sub_dir = data_info['img_info']['filename'].split('/')[0]
            if sub_dir not in self.ct_dict.keys():
                self.ct_dict[sub_dir] = [(data_info['img_info']['filename'] + ' %d' % data_info['gt_label'], idx)]
            else:
                self.ct_dict[sub_dir].append((data_info['img_info']['filename'] + ' %d' % data_info['gt_label'], idx))

        print(len(self.ct_dict) * num_per_ct, "num_per_ct is %d, pos_fraction is %.2f" % (num_per_ct, pos_fraction))
        if self.round_up:
            self.total_size = math.ceil(len(self.ct_dict) * num_per_ct / self.num_replicas) * self.num_replicas
        else:
            self.total_size = len(self.ct_dict) * num_per_ct

    def __iter__(self):
        # deterministically shuffle based on epoch
        self.pos_num = 0
        self.neg_num = 0
        self.lower = 0
        self.zero = 0

        indices = []
        ct_sub_dirs = list(self.ct_dict.keys())

        for sub_dir in ct_sub_dirs:
            pos_list, neg_list = self._split_lines(self.ct_dict[sub_dir])
            sample_list = self._sample_patch(pos_list, neg_list, self.num_per_ct, self.pos_fraction)
            indices.extend(self._get_index(sample_list))

        print("pos sample num is: %d, neg sample num is: %d (approximately)" % (self.pos_num, self.neg_num))
        print("ct with lower sampled pos num: %d, ct with zero pos num: %d" % (self.lower, self.zero))
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        if self.shuffle:
            SEED = self.epoch
            random.seed(SEED)
            random.shuffle(indices)

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if self.round_up:
            assert len(indices) == math.ceil(len(self.ct_dict) * self.num_per_ct / self.num_replicas)

        return iter(indices)

    def _split_lines(self, line_idx_list):
        pos_list = []
        neg_list = []
        for line_idx in line_idx_list:
            label = int(line_idx[0].split(' ')[1])
            if label == 0:
                neg_list.append(line_idx)
            else:
                pos_list.append(line_idx)

        return pos_list, neg_list

    def _sample_patch(self, pos_list, neg_list, sample_num, pos_fraction=0.5):
        pos_per_image = int(round((sample_num * pos_fraction)))
        pos_per_this_image = min(pos_per_image, len(pos_list))
        # Without deterministic SEED
        sampled_pos = random.sample(pos_list, pos_per_this_image)

        if pos_per_this_image < sample_num * pos_fraction:
            self.lower += 1
        if pos_per_this_image == 0:
            self.zero += 1
        self.pos_num += pos_per_this_image
        neg_per_this_image = sample_num - pos_per_this_image
        neg_per_this_image = min(neg_per_this_image, len(neg_list))
        sampled_neg = random.sample(neg_list, neg_per_this_image)
        self.neg_num += neg_per_this_image

        sample_list = sampled_pos + sampled_neg
        random.shuffle(sample_list)

        while len(sample_list) < sample_num:
            end_idx = min((sample_num - len(sample_list)), len(sample_list))
            sample_list = sample_list + sample_list[:end_idx]
        assert len(sample_list) == sample_num

        return sample_list

    @staticmethod
    def _get_index(sample_list):
        return [line[1] for line in sample_list]

