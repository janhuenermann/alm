import torch

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
