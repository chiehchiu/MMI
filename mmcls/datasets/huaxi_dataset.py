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
class huaxiCTDataset(BaseDataset):
    CLASSES = ['慢阻肺', '支扩','气胸','肺气肿','肺炎','间质性肺炎','肺结核','积液','肺癌', '肺大泡']

    def __init__(self, data_prefix, pipeline, feat_path=None, ann_file=None, sub_set=None, test_mode=False, use_sid_sampler=False):
        self.feat_path = feat_path
        super(huaxiCTDataset, self).__init__(data_prefix=data_prefix, pipeline=pipeline, classes=self.CLASSES, ann_file=ann_file, test_mode=test_mode,sub_set=sub_set,use_sid_sampler=use_sid_sampler)

    def load_annotations(self):
        '''Overwrite load_annotations func.
        '''
        ann_file = self.ann_file
        with open(ann_file,'r') as f:
            tmp=csv.reader(f)
            lines = []
            for i in tmp:
                lines.append(i)
        lines=lines[1:]
        data_infos = []
        for index in range(len(lines)):
            ID, path, label = lines[index]
            label = label[1:-1]
            labels = []
            for i in label.split(','):
                labels.append(int(i))
            #filename = os.path.join(self.data_prefix,'data_outputsize_64_256_256', path,'norm_image.npz')
            filename = os.path.join(path,'norm_image.npz')
            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['gt_label'] = np.array((labels), dtype=np.int64)
            data_infos.append(data_info)
        # load feats
        if self.feat_path is None:
            return data_infos
        feat_dict = load_feats(self.feat_path)
        for data_info in data_infos:
            feat_idx = feat_dict['filenames'][data_info['img_info']['filename']]
            feat = feat_dict['feats'][feat_idx, :]
            data_info['aux_feat'] = feat
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
@DATASETS.register_module()
class huaxiMultiPneuCT4Dataset(huaxiCTDataset):
    CLASSES = ['病毒性肺炎', '真菌性肺炎', '细菌性肺炎', '结核性肺炎']

@DATASETS.register_module()
class huaxiMultiPneuCT5Dataset(huaxiCTDataset):
    CLASSES = ['细菌性肺炎', '真菌性肺炎', '病毒性肺炎', '结核性肺炎', '重症肺炎']

@DATASETS.register_module()
class huaxiMultiPneuCTClsDataset(huaxiCTDataset):
    CLASSES = ['非肺炎', '肺炎']

@DATASETS.register_module()
class huaxiMultiPneuCT13to10Dataset(huaxiCTDataset):
    CLASSES = ['乙型流感病毒','偏肺病毒','冠状病毒','副流感病毒','呼吸道合胞病毒','甲型流感病毒','甲流H1N1（2009）','肺炎支原体','腺病毒','鼻病毒']

@DATASETS.register_module()
class huaxiMultiPneuCTnew9Dataset(huaxiCTDataset):
    CLASSES = ['耐甲氧西林葡萄球菌','结核分枝杆菌复合群','嗜麦牙窄食单胞菌','大肠埃希菌','流感嗜血杆菌','肺炎克雷伯菌','金黄色葡萄球菌','铜绿假单胞菌','鲍曼不动杆菌']

@DATASETS.register_module()
class huaxiMultiPneCriticalDataset(huaxiCTDataset):
    CLASSES = ['非危重','危重']
