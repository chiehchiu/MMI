import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, multi_label_smooth=False, epsilon=0.1):
    # element-wise losses
    loss = F.cross_entropy(pred, label.float(), reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

# LJ
def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

# LJ
def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         multi_label_smooth=False,
                         epsilon=0.1):
    if multi_label_smooth:
        label = label_smoothing(label, epsilon)
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none') # none
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def label_smoothing(label, epsilon=0.1):
    label = (1-epsilon) * label + epsilon/2
    return label

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0,
                 multi_label_smooth=False,
                 epsilon=0.1):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.multi_label_smooth = multi_label_smooth
        self.epsilon = epsilon

        # LJ
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            multi_label_smooth=self.multi_label_smooth,
            epsilon=self.epsilon,
            **kwargs)
        return loss_cls
