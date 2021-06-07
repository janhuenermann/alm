import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional

from alm.geometry.polygon import area_of_intersection, shoelace


@torch.jit.script
def box_iou(boxes1, boxes2, strict: bool = False, eps: float = 0.):
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
def generalized_box_iou(boxes1, boxes2, strict: bool = False, eps: float = 0.):
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
def convex_iou(poly1, poly2, eps: float = 0.):
    """
    Intersection over Union (IoU) of two ccw-oriented
    convex polygons.

    poly1: [*, n, 2]
    poly2: [*, m, 2]
    ---
    returns: [*]
    """
    area1 = shoelace(poly1)
    area2 = shoelace(poly2)
    inter = area_of_intersection(poly1, poly2)
    union = area1 + area2 - inter
    return inter / (union + eps)


@torch.jit.script
def rotation_basis(angle):
    c, s = angle.cos(), angle.sin()
    return torch.stack((c, -s, s, c), -1).view(angle.shape + (2, 2,))


@torch.jit.script
def xywha_to_4xy(xywha):
    T = torch.tensor([
        [-1, -1],
        [-1,  1],
        [ 1,  1],
        [ 1, -1]], device=xywha.device, dtype=xywha.dtype)
    basis = xywha[..., 2:4, None] / 2. * rotation_basis(xywha[..., 4])
    return xywha[..., :2] + torch.matmul(T.expand(basis.shape[:-2] + (-1, -1,)), basis)


@torch.jit.script
def iou(labels, predictions, tensor_format: str = "xyxy"):
    """
    Returns the IOU of label and prediction. The format of the bounding boxes
    can be specified with the bbox_format parameter, which is either
    'xyxy' or 'xywha' (includes rotation angle).

    labels: Tensor of shape [*, 4 or 5]
    predictions: Tensor of shape [*, 4 or 5]
    bbox_format: Either 'xyxy' or 'xywha'

    Returns IOU in shape [*]
    """
    assert predictions.size(-1) == labels.size(-1)
    if tensor_format == "xyxy":
        assert labels.size(-1) == 4
        return box_iou(labels, predictions)
    elif tensor_format == "xywha":
        assert labels.size(-1) == 5
        return convex_iou(xywha_to_4xy(labels), xywha_to_4xy(predictions))
    else:
        raise RuntimeError("Unrecognized tensor format: {}".format(tensor_format))


@torch.jit.script
def compute_true_positives(labels, predictions, confidence,
                           labels_mask: Optional[Tensor] = None,
                           predictions_mask: Optional[Tensor] = None,
                           iou_threshold: float = 0.5,
                           tensor_format: str = "xyxy"):
    """
    Returns tuple of trues (shape [*, m]), positives (shape [*]), and confidence (sorted, shape [*, m]).

    labels: Ground truth bounding boxes with shape [*, n, 4]
    predictions: Predicted bounding boxes with shape [*, m, 4]
    confidence: Confidence of each prediction with shape [*, m]
    labels_mask: Boolean mask of labels with shape [*, n]
    predictions_mask: Boolean mask of predictions with shape [*, m]
    iou_threshold: Threshold of IOU. Predictions with IOU > threshold to labels will be regarded as true positives
    tensor_format: Format of predictions/labels, either 'xyxy' or 'xywha'
    """
    confidence, perm = confidence.sort(-1, descending=True)
    predictions = predictions.gather(-2, perm.unsqueeze(-1).expand_as(predictions))

    if predictions_mask is not None:
        predictions_mask = predictions_mask.gather(-1, perm)

    _ious = iou(labels[..., :, None, :], predictions[..., None, :, :], tensor_format=tensor_format)
    if labels_mask is not None:
        _ious.masked_fill_(~labels_mask.unsqueeze(-1), 0.)
    if predictions_mask is not None:
        _ious.masked_fill_(~predictions_mask.unsqueeze(-2), 0.)
    _ious.nan_to_num_(0., posinf=0., neginf=0.)

    trues = _ious >= iou_threshold # [*, n, m]

    matched = trues.cummax(-1)[0].roll(1, -1)
    matched[..., :, 0] = False

    trues = (trues & (~matched)).max(-2, False)[0] # [*, m]

    if labels_mask is not None:
        positives = labels_mask.sum(-1, True)
    else:
        positives = torch.tensor(labels.size(-1), dtype=labels.dtype, device=labels.device)

    return trues, positives, confidence


@torch.jit.script
def precision_recall_from_tp(trues, positives, confidence: Optional[Tensor] = None,
                             image_dim: Optional[int] = None):
    """
    Computes the precision recall curve from
    trues (shape [*, m]), positive counts (shape [*]),
    and confidence (sorted, shape [*, m]).

    trues: Boolean tensor with shape [*, m]
    positives: Count of positives of shape [*]
    confidence: Confidence of each prediction with shape [*, m]
    image_dim: If not none, merges predictions over multiple images together
    """
    if image_dim is not None:
        assert confidence is not None
        if image_dim < 0:
            image_dim += confidence.dim()
        assert image_dim < confidence.dim() - 1, "Image dim ({}) cannot overlap with bbox dim".format(image_dim)
        if positives.dim() > 0:
            assert positives.dim() == trues.dim()
            positives = positives.transpose(image_dim, -2).flatten(-2).sum(-1, True)
        else:
            assert positives.dim() == 0
            positives *= int(trues.size(image_dim))
        confidence, perm = confidence.transpose(image_dim, -2).flatten(-2).sort(-1, descending=True)
        trues = trues.transpose(image_dim, -2).flatten(-2).gather(-1, perm)

    tp = trues.float().cumsum(-1)
    tp_plus_fp = torch.arange(1, tp.size(-1) + 1, device=tp.device, dtype=tp.dtype).expand_as(tp)

    recall = tp / positives
    precision = tp / tp_plus_fp

    return precision, recall


@torch.jit.script
def average_precision_from_pr(precision, recall, interpolation_points: Optional[int] = None):
    """
    Computes the average precision (AP) from precision and recall.

    interpolation_points: Optional integer specifying the number of interpolation points (as in VOC Pascal)

    Returns AP tensor with same shape as precision and recall, but with the last dimension reduced.
    """
    if interpolation_points is not None:
        recall_points = torch.linspace(0., 1., interpolation_points, device=precision.device, dtype=precision.dtype)\
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


@torch.jit.script
def precision_recall(labels, predictions, confidence,
                     labels_mask: Optional[Tensor] = None,
                     predictions_mask: Optional[Tensor] = None,
                     image_dim: Optional[int] = None,
                     iou_threshold: float = 0.5,
                     tensor_format: str = "xyxy"):
    """
    Computes the precision-recall curve (P-R curve) given
    predictions and labels. If `image_dim` is set, aggregates
    over multiple images.
    
    labels: Ground truth bounding boxes with shape [*, n, 4]
    predictions: Predicted bounding boxes with shape [*, m, 4]
    confidence: Confidence of each prediction with shape [*, m]
    labels_mask: Boolean mask of labels with shape [*, n]
    predictions_mask: Boolean mask of predictions with shape [*, m]
    image_dim: Dimension of each image (e.g. -2); -1 refers to bbox dimension, therefore it must be different
    iou_threshold: Threshold of IOU. Predictions with IOU > threshold to labels will be regarded as true positives
    tensor_format: Format of predictions/labels, either 'xyxy' or 'xywha'
    """
    trues, positives, confidence = compute_true_positives(
        labels, predictions, confidence, labels_mask, predictions_mask, iou_threshold, tensor_format)

    return precision_recall_from_tp(trues, positives, confidence, image_dim)


@torch.jit.script
def average_precision(labels, predictions, confidence,
                      labels_mask: Optional[Tensor] = None,
                      predictions_mask: Optional[Tensor] = None,
                      image_dim: Optional[int] = None,
                      iou_threshold: float = 0.5,
                      interpolation_points: Optional[int] = None,
                      tensor_format: str = "xyxy"):
    """
    Computes the average precision (AP) given
    predictions and labels. If `image_dim` is set, aggregates
    over multiple images.
    
    labels: Ground truth bounding boxes with shape [*, n, 4]
    predictions: Predicted bounding boxes with shape [*, m, 4]
    confidence: Confidence of each prediction with shape [*, m]
    labels_mask: Boolean mask of labels with shape [*, n] (True for existance of label)
    predictions_mask: Boolean mask of predictions with shape [*, m] (True for existance of prediction)
    image_dim: Dimension of each image (e.g. -2); -1 refers to bbox dimension, therefore it must be different
    iou_threshold: Threshold of IOU. Predictions with IOU > threshold to labels will be regarded as true positives
    interpolation_points: Number of interpolation points of P-R-Curve. If None, will not interpolate and integrate whole curve (default).
    tensor_format: Format of predictions/labels, either 'xyxy' or 'xywha'
    """
    precision, recall = precision_recall(labels, predictions, confidence,
        labels_mask, predictions_mask, image_dim, iou_threshold, tensor_format)

    return average_precision_from_pr(precision, recall)
