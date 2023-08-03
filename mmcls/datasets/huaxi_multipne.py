from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import random
import os
import numpy as np
import pandas as pd
from .builder import DATASETS
import csv
import pdb
import pickle

def load_feats(result_path):
    result_dict = pickle.load(open(result_path, 'rb'))
    file_list = result_dict['filenames']
    file_dict = {}
    for idx, filename in enumerate(file_list):
        file_dict[filename] = idx
    result_dict['filenames'] = file_dict
    return result_dict

@DATASETS.register_module()
class huaxiMultipneDataset(BaseDataset):
    CLASSES = ['细菌性肺炎', '真菌性肺炎', '病毒性肺炎', '结核性肺炎']
    def __init__(self, data_prefix, pipeline, feat_path=None, ann_file=None, sub_set=None, test_mode=False, use_sid_sampler=False):
        self.feat_path = feat_path
        super(huaxiMultipneDataset, self).__init__(data_prefix=data_prefix, pipeline=pipeline, classes=self.CLASSES, ann_file=ann_file, test_mode=test_mode,sub_set=sub_set,use_sid_sampler=use_sid_sampler)

    def load_annotations(self):
        ann_file = self.ann_file #TODO
        f = open(ann_file,'rb')
        subset = pickle.load(f)
        f.close()
        data_infos = []
        for index in subset.keys():
            data_info = {}
            label = subset[index]['label']
            filename = os.path.join(index,'norm_image.npz')
            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['pdesc'] = subset[index]['pdesc']
            data_info['bics'] = subset[index]['bics']
            data_info['bts'] = subset[index]['bts']
            data_info['gt_label'] = np.array((label), dtype=np.int64)
            data_infos.append(data_info)
        return data_infos

    def __len__(self):
        if self.use_sid_sampler:
            sub_dirs = [data_info['img_info']['filename'] for data_info in self.data_infos]
            sids = ['/'.join(sub_dir.split('/')[:2]) for sub_dir in sub_dirs]
            sids = list(set(sids))
            length = len(sids)
        else:
            length = len(self.data_infos)

        return length

