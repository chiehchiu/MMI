from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import random
import os
import numpy as np
import pandas as pd
from .builder import DATASETS

@DATASETS.register_module()
class AneurysmDataset(BaseDataset):

    CLASSES = ['fp', 'aneurysm']

    def load_annotations(self):
        '''Overwrite load_annotations func.
        '''
        ann_file = os.path.join(self.data_prefix, self.ann_file)
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        lines =  [line.strip().split() for line in lines]  # add support for [' ' '\t']
        self.eval_patch_info_list = lines
        data_infos = []
        for index in range(len(lines)):
            path, label = lines[index]
            filename = path
            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['gt_label'] = np.array(float(label), dtype=np.int64)
            data_infos.append(data_info)
        return data_infos




