# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from typing import List

import mmcv
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy, cross_entropy, binary_cross_entropy
from .pipelines import Compose

from mmcls.models.losses import get_sensitivity, get_specificity, get_precision, get_F1, get_accuracy, get_aneu_eval_auc_indicator
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, \
            hamming_loss, accuracy_score, roc_curve, auc, confusion_matrix


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 sub_set=None,
                 use_sid_sampler=False):
        super(BaseDataset, self).__init__()
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()
        self.use_sid_sampler = use_sid_sampler
        self.sub_set = sub_set

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.data_infos[idx]['gt_label'])]

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support', \
            'auc', 'all', 'auc_multi_cls', 'extend_aneurysm', 'predict'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        if 'auc' in metrics:
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 5)
            auc = roc_auc_score(gt_labels[:,1], results[:,1])
            eval_results['loss'] = loss
            eval_results['auc'] = auc
            pred_labels = np.argmax(results, 1)
            print("confusion matrix:")
            print(confusion_matrix(pred_labels, gt_labels[:,1]))
        if'all' in metrics:
            # gt_labels.shape (n,) n is num of samples
            # results.shape(n, 2) 2 is n_classes
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 3)
            auc = roc_auc_score(gt_labels, results[:,1])
            #acc = accuracy(results, gt_labels, topk)
            #prec, rec, f1, _ = precision_recall_fscore_support(gt_labels, results[:, 1].round(), average="binary")
            #specificity = get_specificity(torch.Tensor(results[:,1]), torch.Tensor(gt_labels))
            prec, rec, f1, specificity, acc = self.get_best_metrics(gt_labels, results[:, 1])


            #eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            eval_results['loss'] = loss
            eval_results['acc'] = acc
            eval_results['auc'] = auc
            eval_results['f1'] = f1
            eval_results['recall'] = rec
            eval_results['precision'] = prec
            eval_results['specificity'] = specificity
        # LJ
        if 'auc_multi_cls' in metrics:
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            # 训练时的loss为每个样本的loss 评估时为每个label的loss
            # train: loss.sum() / sample_num  val: loss.mean()
            # loss = binary_cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = F.binary_cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels)).mean()
            loss = float(loss)
            eval_results['loss'] = loss

            auc_pre_cls = {}
            auc_total = 0.0
            for cls in self.CLASSES:
                idx = self.CLASSES.index(cls)
                if gt_labels[:,idx].sum() == 0:
                    _auc = 1.0
                else:
                    _auc = roc_auc_score(gt_labels[:, idx], results[:, idx])
                # fpr, tpr, thresholds = roc_curve(gt_labels[:, idx], results[:, idx], pos_label=1)
                # _auc = auc(fpr, tpr)
                auc_total += _auc
                auc_pre_cls[cls] = round(_auc, 4)
            # eval_results['auc_mean'] = roc_auc_score(gt_labels, results)
            eval_results['auc_mean'] = auc_total / len(self.CLASSES)
            eval_results['auc_pre_cls'] = auc_pre_cls
        # predict result
        if 'predict' in metrics:
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            #predict_cls = ['ct\tground truth\tprediction']
            predict_cls = []
            for sample_i,data in enumerate(self.data_infos):
                gt_label = str([1 if l>=0.5 else 0 for l in data['gt_label']])
                ct_triple = '/'.join(data['img_info']['filename'].split('/')[:-1])
                pd_label = str([pd_prob for pd_prob in results[sample_i]])
                pred_str = '\t'.join([ct_triple,gt_label,pd_label])
                predict_cls.append(pred_str)
            eval_results['predict'] = predict_cls
        if 'extend_aneurysm' in metrics: # slice & lesion level auc
            # gt_labels.shape (n,) n is num of samples
            # results.shape(n, 2) 2 is n_classes
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 4)
            auc = roc_auc_score(gt_labels, results[:,1])
            #acc = accuracy(results, gt_labels, topk)
            #prec, rec, f1, _ = precision_recall_fscore_support(gt_labels, results[:, 1].round(), average="binary")
            #specificity = get_specificity(torch.Tensor(results[:,1]), torch.Tensor(gt_labels))
            prec, rec, f1, specificity, acc = self.get_best_metrics(gt_labels, results[:, 1].copy())
            seg_auc_info_list, other_infos = get_aneu_eval_auc_indicator(results[:, 1], self.eval_patch_info_list)


            #eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            eval_results['loss'] = loss
            eval_results['acc'] = acc
            eval_results['auc'] = auc
            eval_results['f1'] = f1
            eval_results['recall'] = rec
            eval_results['precision'] = prec
            # eval_results['specificity'] = specificity
            # eval_results['seg-auc'] = "({})".format(';'.join(seg_auc_info_list))
            # eval_results['seg-other'] = "({})".format(';'.join(other_infos))
            for eval_info in seg_auc_info_list:
                key, value = eval_info.split('=', maxsplit=1)
                eval_results[key] = value
            for eval_info in other_infos:
                key, value = eval_info.split('=', maxsplit=1)
                eval_results[key] = value
        return eval_results

    def get_best_metrics(self, gts, preds):
        thresh_array = np.arange(0,1,0.1)
        res = 0
        best_thresh = 0
        for thresh in thresh_array:
            specificity = get_specificity(torch.Tensor(preds), torch.Tensor(gts), thresh)
            sensitivity = get_sensitivity(torch.Tensor(preds), torch.Tensor(gts), thresh)
            tmp_metric = specificity + sensitivity
            tmp = (specificity * sensitivity) / (specificity + sensitivity)
            if tmp_metric > res:
                res = tmp_metric
                best_thresh = thresh
            #print('thre, spe, sen, sum, f1 {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(thresh, specificity,sensitivity, tmp_metric, tmp))
        specificity = get_specificity(torch.Tensor(preds), torch.Tensor(gts), best_thresh)
        acc = get_accuracy(torch.Tensor(preds), torch.Tensor(gts), best_thresh)
        preds[preds > best_thresh] = 1.0
        preds[preds <= best_thresh] = 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(gts, preds, average="binary")
        return prec, rec, f1, specificity, acc
