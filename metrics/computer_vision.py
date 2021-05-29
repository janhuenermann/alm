import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

@torch.jit.script
def iou(boxes1, boxes2, strict: bool = False, eps: float = 0.):
   """
   Intersection over Union (IoU).

   boxes1: Boxes with shape [*, 4] in format x1 y1 x2 y2
   boxes2: Boxes with shape [*, 4] in format x1 y1 x2 y2
   strict: Bool whether to check if x1 > x2 or y1 > y2
   eps: Numerical stability epsilon used for division
   """
   mins = torch.min(boxes1, boxes2)
   maxs = torch.max(boxes1, boxes2)

   wh1 = boxes1[..., 2:] - boxes1[..., :2]
   wh2 = boxes2[..., 2:] - boxes2[..., :2]
   whi = mins[..., 2:]   - maxs[..., :2]

   if strict:
    wh1.clamp_(min=0)
    wh2.clamp_(min=0)
   whi.clamp_(min=0)

   area1 = wh1.prod(-1)
   area2 = wh2.prod(-1)
   inter = whi.prod(-1)
   union = area1 + area2 - inter
   return inter / (union + eps)


# From https://gist.github.com/janhuenermann/9410805647a3fb1521f1b754360eeefe
@torch.jit.script
def generalized_iou(boxes1, boxes2, strict: bool = False, eps: float = 0.):
   """
   Generalized IoU (https://giou.stanford.edu/)

   boxes1: Boxes with shape [*, 4] in format x1 y1 x2 y2
   boxes2: Boxes with shape [*, 4] in format x1 y1 x2 y2
   strict: Bool whether to check if x1 > x2 or y1 > y2
   eps: Numerical stability epsilon used for division
   """
   mins = torch.min(boxes1, boxes2)
   maxs = torch.max(boxes1, boxes2)

   wh1 = boxes1[..., 2:] - boxes1[..., :2]
   wh2 = boxes2[..., 2:] - boxes2[..., :2]
   whi = mins[..., 2:] - maxs[..., :2]
   who = maxs[..., 2:] - mins[..., :2]

   if strict:
      for wh in (wh1, wh2, who):
         wh.clamp_(min=0)
   whi.clamp_(min=0)

   area1, area2 = wh1.prod(-1), wh2.prod(-1)
   inter, outer = whi.prod(-1), who.prod(-1)
   union = area1 + area2 - inter

   return inter / (union + eps) + union / (outer + eps) - 1.


@torch.jit.script
def precision_recall(labels, sorted_predictions,
                     labels_mask: Optional[Tensor] = None,
                     predictions_mask: Optional[Tensor] = None,
                     iou_threshold: float = 0.5):
   """
   Computes the precision-recall curve (P-R curve) given
   a ranked list of detections (decreasing probability)
   and labels.
   
   labels: Ground truth bounding boxes with shape [*, n, 4]
   mask: Booleann mask of labels with shape [*, n]
   sorted_predictions: Prediction bounding boxes, sorted, with shape [*, m, 4]
   iou_threshold: Threshold of IOU when to count detection as true positive
   """
   m = sorted_predictions.size(-2)

   _ious = iou(labels[..., :, None, :], sorted_predictions[..., None, :, :])
   if labels_mask is not None:
      _ious.masked_fill_(~labels_mask.unsqueeze(-1), 0.)
   if predictions_mask is not None:
      _ious.masked_fill_(~predictions_mask.unsqueeze(-2), 0.)
   _ious.nan_to_num_(0., posinf=0., neginf=0.)

   trues = _ious >= iou_threshold

   matched = trues.cummax(-1)[0].roll(1, -1)
   matched[..., :, 0] = False

   trues = (trues & (~matched)).max(-2, False)[0]
   falses = ~trues

   tp = trues.float().cumsum(-1)
   fp = falses.float().cumsum(-1)

   if labels_mask is not None:
      positives = labels_mask.sum(-1, True)
   else:
      positives = torch.tensor(labels.size(-1), dtype=labels.dtype, device=labels.device)

   recall = tp / positives
   precision = (tp / (tp + fp))

   return precision, recall


@torch.jit.script
def average_precision(labels, sorted_predictions,
                      labels_mask: Optional[Tensor] = None,
                      predictions_mask: Optional[Tensor] = None,
                      iou_threshold: float = 0.5, interpolation_points: Optional[int] = None):
   """
   Computes the average precision (AP) given
   a ranked list of detections (decreasing probability)
   and labels.
   
   labels: Ground truth bounding boxes with shape [*, n, 4]
   mask: Booleann mask of labels with shape [*, n]
   sorted_predictions: Prediction bounding boxes, sorted, with shape [*, m, 4]
   iou_threshold: Threshold of IOU when to count detection as true positive
   interpolation_points: Number of interpolation points of P-R-Curve
   """
   precision, recall = precision_recall(labels, sorted_predictions, labels_mask, predictions_mask, iou_threshold)

   if interpolation_points is not None:
      recall_points = torch.linspace(0., 1., interpolation_points, device=labels.device, dtype=labels.dtype)\
         .expand(recall.shape + (interpolation_points,))
      interpolated = (precision[..., None] * (recall[..., None] >= recall_points).float())\
         .max(-2, False)[0]
      result = interpolated.mean(-1)
   else:
      precision = precision.flip(-1).cummax(-1)[0].flip(-1)
      area_under_curve = torch.diff(recall, dim=-1) * precision[..., 1:]
      area_under_curve = area_under_curve.sum(-1) + recall[..., 0] * precision[..., 0]
      result = area_under_curve

   return result
