import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Optional
import math

from alm.utils import check_shape
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
    assert check_shape(boxes1, [None, 4], "boxes1")
    assert check_shape(boxes2, [None, 4], "boxes2")

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
    assert check_shape(boxes1, [None, 4], "boxes1")
    assert check_shape(boxes2, [None, 4], "boxes2")

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
    assert check_shape(poly1, [None, -1, 2], "poly1")
    assert check_shape(poly2, [None, -1, 2], "poly2")
    fut_inter = torch.jit.fork(area_of_intersection, poly1, poly2)
    area1 = shoelace(poly1)
    area2 = shoelace(poly2)
    inter = torch.jit.wait(fut_inter)
    union = area1 + area2 - inter
    return inter / (union + eps)


@torch.jit.script
def rotation_basis(angle, transpose: bool = False):
    c, s = angle.cos(), angle.sin()
    if transpose:
        out = torch.stack((c, s, -s, c), -1)
    else:
        out = torch.stack((c, -s, s, c), -1)
    return out.view(angle.shape + (2, 2,))


@torch.jit.script
def xywh_to_xy4(xywh):
    assert check_shape(xywh, [None, 4], "xywh")
    xy, wh = xywh[..., None, :2], xywh[..., 2:4]
    T = torch.tensor([[-0.5, -0.5], [-0.5,  0.5], [ 0.5,  0.5], [ 0.5, -0.5]])
    T = T.to(wh).expand(wh.shape[:-1] + (-1, -1,))
    return xy + T * wh.unsqueeze(-2)  # broadcast along row dimension


@torch.jit.script
def xywha_to_xy4(xywha, angle: Optional[Tensor] = None, upper_left_first: bool = False):
    assert check_shape(xywha, [None, 5], "xywha")
    if angle is None:
        angle = xywha[..., 4]
    xy, basis_T = xywha[..., None, :2], xywha[..., 2:4, None] * rotation_basis(angle, transpose=True)
    T = torch.tensor([[-0.5, -0.5], [-0.5,  0.5], [ 0.5,  0.5], [ 0.5, -0.5]])
    T = T.to(basis_T).expand(basis_T.shape[:-2] + (-1, -1,))
    points = xy + T.matmul(basis_T)
    if upper_left_first:
        shifts = torch.round(2. * angle.detach() / math.pi).long()
        indices = torch.arange(0, 4).to(points.device).expand(shifts.shape + (4,)) + shifts.unsqueeze(-1)
        points = points.gather(-2, (indices % 4).unsqueeze(-1).expand(indices.shape + (2,)))
    return points


@torch.jit.script
def project_rotated_boxes(xywha1, xywha2, upper_left_first: bool = False):
    """
    Projects rotated bounding box xywha2 to the local coordinate system
    of rotated bounding box xywha1 and returns the polygon.

    Returns
    -------
    poly1
        Tensor in shape (*, 4, 2) with counter-clockwise polygon coords
        of projected bounding box xywha1 (always a rectangle)
    poly2
        Tensor in shape (*, 4, 2) with counter-clockwise polygon coords
        of projected bounding box xywha2
    """
    assert check_shape(xywha1, [None, 5], "xywha1")
    assert check_shape(xywha2, [None, 5], "xywha2")
    T = torch.tensor([[-0.5, -0.5], [-0.5,  0.5], [ 0.5,  0.5], [ 0.5, -0.5]]).to(xywha1)
    # Project 1
    proj1 = T.expand(xywha1.shape[:-1] + (-1, -1,)) * xywha1[..., 2:4].unsqueeze(-2)
    # Project 2
    offset = (xywha2[..., None, :2] - xywha1[..., None, :2]).matmul(rotation_basis(-xywha1[..., 4], transpose=True))
    angle = xywha2[..., 4] - xywha1[..., 4]
    basis2_T = xywha2[..., 2:4, None] * rotation_basis(angle, transpose=True)
    proj2 = T.expand(xywha2.shape[:-1] + (-1, -1,)).matmul(basis2_T) + offset
    if upper_left_first:
        shifts = torch.round(2. * angle.detach() / math.pi).long()
        indices = torch.arange(0, 4).to(proj2.device).expand(shifts.shape + (4,)) + shifts.unsqueeze(-1)
        proj2 = proj2.gather(-2, (indices % 4).unsqueeze(-1).expand(indices.shape + (2,)))
    return proj1, proj2


@torch.jit.script
def xy4_to_box(xy4):
    assert check_shape(xy4, [None, 2], "xy4")
    return torch.cat((xy4.min(-2)[0], xy4.max(-2)[0]), -1)


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
    if tensor_format == "xyxy":
        return box_iou(labels, predictions)
    elif tensor_format == "xywha":
        return convex_iou(xywha_to_xy4(labels), xywha_to_xy4(predictions))
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
