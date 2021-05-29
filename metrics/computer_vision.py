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
def precision_recall(labels, predictions, confidence,
                     labels_mask: Optional[Tensor] = None,
                     predictions_mask: Optional[Tensor] = None,
                     image_dim: Optional[int] = None,
                     iou_threshold: float = 0.5):
   """
   Computes the precision-recall curve (P-R curve) given
   a ranked list of detections (decreasing probability)
   and labels.
   
   labels: Ground truth bounding boxes with shape [*, n, 4]
   predictions: Predicted bounding boxes with shape [*, m, 4]
   confidence: Confidence of each prediction with shape [*, m]
   labels_mask: Boolean mask of labels with shape [*, n]
   predictions_mask: Boolean mask of predictions with shape [*, m]
   image_dim: Dimension of each image (e.g. -2); -1 refers to bbox dimension, therefore it must be different
   iou_threshold: Threshold of IOU. Predictions with IOU > threshold to labels will be regarded as true positives
   """
   m = predictions.size(-2)
   n = int(labels.size(-1))

   conf_perm = confidence.argsort(-1, descending=True)

   _ious = iou(labels[..., :, None, :], predictions.gather(-2, conf_perm.unsqueeze(-1).expand_as(predictions))[..., None, :, :])
   if labels_mask is not None:
      _ious.masked_fill_(~labels_mask.unsqueeze(-1), 0.)
   if predictions_mask is not None:
      _ious.masked_fill_(~predictions_mask.gather(-1, conf_perm).unsqueeze(-2), 0.)
   _ious.nan_to_num_(0., posinf=0., neginf=0.)

   trues = _ious >= iou_threshold # [*, n, m]

   matched = trues.cummax(-1)[0].roll(1, -1)
   matched[..., :, 0] = False

   trues = (trues & (~matched)).max(-2, False)[0] # [*, m]

   if image_dim is not None:
      if image_dim < 0:
         image_dim += confidence.dim()
      assert image_dim < confidence.dim() - 1, "Image dim ({}) cannot overlap with bbox dim".format(image_dim)
      trues = trues.transpose(image_dim, -2).flatten(-2)
      all_confidences = confidence.gather(-1, conf_perm).transpose(image_dim, -2).flatten(-2)
      trues = trues.gather(-1, all_confidences.argsort(-1, descending=True))
      n *= int(labels.size(image_dim))
      if labels_mask is not None:
         labels_mask = labels_mask.transpose(image_dim, -2).flatten(-2)

   falses = ~trues

   tp = trues.float().cumsum(-1)
   fp = falses.float().cumsum(-1)

   if labels_mask is not None:
      positives = labels_mask.sum(-1, True)
   else:
      positives = torch.tensor(n, dtype=labels.dtype, device=labels.device)

   recall = tp / positives
   precision = (tp / (tp + fp))

   return precision, recall


@torch.jit.script
def average_precision(labels, predictions, confidence,
                      labels_mask: Optional[Tensor] = None,
                      predictions_mask: Optional[Tensor] = None,
                      image_dim: Optional[int] = None,
                      iou_threshold: float = 0.5,
                      interpolation_points: Optional[int] = None):
   """
   Computes the average precision (AP) given
   a ranked list of detections (decreasing probability)
   and labels.
   
   labels: Ground truth bounding boxes with shape [*, n, 4]
   predictions: Predicted bounding boxes with shape [*, m, 4]
   confidence: Confidence of each prediction with shape [*, m]
   labels_mask: Boolean mask of labels with shape [*, n] (True for existance of label)
   predictions_mask: Boolean mask of predictions with shape [*, m] (True for existance of prediction)
   image_dim: Dimension of each image (e.g. -2); -1 refers to bbox dimension, therefore it must be different
   iou_threshold: Threshold of IOU. Predictions with IOU > threshold to labels will be regarded as true positives
   interpolation_points: Number of interpolation points of P-R-Curve. If None, will not interpolate and integrate whole curve (default).
   """
   precision, recall = precision_recall(labels, predictions, confidence, labels_mask, predictions_mask, image_dim, iou_threshold)

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
