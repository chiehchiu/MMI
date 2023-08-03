"""
reference from: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/evaluation.py
"""

import torch
from functools import reduce

import numpy as np
from sklearn.metrics import roc_auc_score

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    #tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    tensor_size = reduce(lambda x,y: x*y, SR.shape)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = (SR==1)&(GT==1)
    FN = (SR==0)&(GT==1)

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    #import pdb
    #pdb.set_trace()
    TN = (SR==0)&(GT==0)
    FP = (SR==1)&(GT==0)

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = (SR==1)&(GT==1)
    FP = (SR==1)&(GT==0)

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_aneu_eval_auc_indicator(pred_score_list, eval_patch_info_list, mtype=('add','mul'), verbose=0):
    """[summary]

    Args:
        pred_score_list ([list]): pred prob list
        eval_patch_info_list ([list]): 2d list of slice info desc and class
        mtype (str, list, optional): [description]. Defaults to 'add'.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        [list]: [description] diff dim of auc info list
    """
    pred_count = len(pred_score_list)
    eval_list_count = len(eval_patch_info_list)
    assert pred_count == eval_list_count, 'aneu eval param length except: {} != {}'.format(pred_count, eval_list_count)
    
    # seg lesion/slice patch props auc calc
    auc_indicator_group_base_slice = {'stage1': [], 'stage2': []}
    auc_indicator_group_base_lesion = {'stage1': {}, 'stage2': {}}
    
    # 融合分组清单
    mtype_list = []
    if isinstance(mtype, (tuple,list)):
        mtype_list = mtype
    else:
        mtype_list = [mtype]
    for _mtype in mtype_list:
        merge_group_name = f'merge-{_mtype}'
        auc_indicator_group_base_slice.setdefault(merge_group_name, [])
        auc_indicator_group_base_lesion.setdefault(merge_group_name, {})
    
    slice_patch_count = {'pos': 0, 'neg': 0, 'all': 0}
    seg_lesion_count = {'pos': set(), 'neg': set(), 'all': set()}
    match_gt_lesion_count = set()
    gt_sample_lesion_count = set()
    case_count = {'all': set(), 'pos': set()}
    
    # stage1 slice patch and lesion match gt info
    for pred_score, (sub_path, _cls) in zip(pred_score_list, eval_patch_info_list):
        #eg: x151_y127_z172_w32_h32_d32_s0.39_seg_neg_5_0.png
        series_name, slice_patch_info = sub_path.split('/')
        *slice_info, seg_gt_label, pos_neg_label, inst_index, recall_inst_index = slice_patch_info.split('_')
        
        # all
        slice_patch_index = sub_path
        lesion_index = f'{series_name};{inst_index}'
        
        case_count['all'].add(series_name)
        
        if 'gt' == seg_gt_label:
            gt_lesion_key = f'{series_name};{inst_index}' # var:inst_index same to var:recall_inst_index
            gt_sample_lesion_count.add(gt_lesion_key)
            case_count['pos'].add(series_name)
            # continue
        elif 'seg' == seg_gt_label:
            s1_score = float(slice_info[-1][1:])
            s2_score = pred_score
            # 1. slice patch level of seg props only
            slice_type = int(_cls)
            auc_indicator_group_base_slice['stage1'] += [(slice_patch_index, s1_score, slice_type)]
            auc_indicator_group_base_slice['stage2'] += [(slice_patch_index, s2_score, slice_type)]          
            
            # 2. lesion level of seg props only            
            lesion_recall_type = int(_cls)
            # 2.1 lesion in stage1
            auc_indicator_group_base_lesion['stage1'].setdefault(lesion_index, [-1, lesion_recall_type])
            cur_lesion_score_s1 = auc_indicator_group_base_lesion['stage1'][lesion_index][0]
            auc_indicator_group_base_lesion['stage1'][lesion_index][0] = max(cur_lesion_score_s1, s1_score)
            
            auc_indicator_group_base_lesion['stage2'].setdefault(lesion_index, [-1, lesion_recall_type])
            cur_lesion_score_s2 = auc_indicator_group_base_lesion['stage2'][lesion_index][0]
            auc_indicator_group_base_lesion['stage2'][lesion_index][0] = max(cur_lesion_score_s2, s2_score)
            
            # 3.merge of slice patch & lesion level
            for _mtype in mtype_list:
                merge_group_name = f'merge-{_mtype}'
                if 'add' == _mtype:
                    score_merge_s1_s2 = (s1_score + s2_score)/2
                elif 'mul' == _mtype:
                    score_merge_s1_s2 = float(np.sqrt(s1_score * s2_score))
                else:
                    raise Exception('unsport merge method for stage1 stage2: {}'.format(_mtype))
                auc_indicator_group_base_slice[merge_group_name] += [(slice_patch_index, score_merge_s1_s2, slice_type)]
                
                auc_indicator_group_base_lesion[merge_group_name].setdefault(lesion_index, [-1, lesion_recall_type])
                cur_lesion_score_ms1s2 = auc_indicator_group_base_lesion[merge_group_name][lesion_index][0]
                auc_indicator_group_base_lesion[merge_group_name][lesion_index][0] = \
                                                                    max(cur_lesion_score_ms1s2, score_merge_s1_s2)
                                                                    
            # 4.count
            slice_patch_count['all'] += 1
            slice_patch_count[pos_neg_label] += 1
            
            seg_lesion_key = f'{series_name};{inst_index}'
            seg_lesion_count['all'].add(seg_lesion_key)
            seg_lesion_count[pos_neg_label].add(seg_lesion_key)
            
            if 'pos' == pos_neg_label:
                recall_inst_index_list = recall_inst_index.rsplit('.png', maxsplit=1)[0]
                recall_inst_index_list = recall_inst_index_list.split('&')
                for recall_inst_index in recall_inst_index_list:
                    match_gt_lesion_key = f'{series_name};{recall_inst_index}'
                    match_gt_lesion_count.add(match_gt_lesion_key)
        else:
            raise Exception('unknown origin type: {} in {}'.format(seg_gt_label, sub_path))

    seg_auc_info_list = []
    for group_name, seg_slice_info in auc_indicator_group_base_slice.items():
        pred_score_list = [_[1] for _ in seg_slice_info]
        label_list = [_[2] for _ in seg_slice_info]
        y_pred_list = np.array(pred_score_list)
        y_gt_list = np.array(label_list)
        auc_score = roc_auc_score(y_true=y_gt_list, y_score=y_pred_list)
        info = f'{group_name}-slice-auc={auc_score:.4f}'
        seg_auc_info_list += [info]
        
    for group_name, seg_lesion_info_map in auc_indicator_group_base_lesion.items():
        seg_lesion_info = list(seg_lesion_info_map.values())
        pred_score_list = [_[0] for _ in seg_lesion_info]
        label_list = [_[1] for _ in seg_lesion_info]
        y_pred_list = np.array(pred_score_list)
        y_gt_list = np.array(label_list)
        auc_score = roc_auc_score(y_true=y_gt_list, y_score=y_pred_list)
        info = f'{group_name}-lesion-auc={auc_score:.4f}'
        seg_auc_info_list += [info]
    
    # count show
    used_seg_slice_count = slice_patch_count
    used_seg_lesion_count_map = {k:len(v) for k, v in seg_lesion_count.items()}
    used_seg_macth_gt_lesion_count = len(match_gt_lesion_count)  # max match gt lesion 
    used_gt_lesion_count = len(gt_sample_lesion_count)  # all gt lesion
    case_total = len(case_count['all']) # all case count
    case_positive = len(case_count['pos'])
    case_negative = case_total - case_positive
    other_infos = [
        f'seg-slice-count={used_seg_slice_count}', 
        f'seg-max-match-gt-lesion={used_seg_macth_gt_lesion_count}/{used_gt_lesion_count}',
        'seg-lesion-count={}=P{}+N{}'.format(*[used_seg_lesion_count_map[k] for k in ['all', 'pos', 'neg']]),
        'eval-case-count={}=P{}+N{}'.format(case_total, case_positive, case_negative),
    ]
    if verbose != 0:
        print('slice count in seg:', used_seg_slice_count)
        print('lesion count in seg:', used_seg_lesion_count_map)
        print('match gt lesion total:', used_seg_macth_gt_lesion_count, used_gt_lesion_count)
    
    return seg_auc_info_list, other_infos
