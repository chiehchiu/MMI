import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, ),
                 # LJ
                 multi_cls=False,
                 dropout_rate=0.,):
        super(LinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = None

        if dropout_rate > 0:
            self.dropout = nn.Dropout3d(p=dropout_rate)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # LJ
        self.multi_cls = multi_cls
        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        # LJ
        if self.multi_cls:
            pred = torch.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, self.multi_cls)

        return losses

    def mixup_forward_train(self, x, gt_a, gt_b, lam):
        cls_score = self.fc(x)
        if self.dropout is not None:
            cls_score = self.dropout(cls_score)
        loss1 = self.loss(cls_score, gt_a, self.multi_cls)
        loss2 = self.loss(cls_score, gt_b, self.multi_cls)
        losses = dict()
        losses['loss'] = lam * loss1['loss'] + (1-lam) * loss2['loss']
        for key in loss1.keys():
            if key != 'loss':
                losses[key] = loss1[key]

        return losses

