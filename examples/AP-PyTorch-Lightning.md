### Object Detection Metrics in PyTorch Lightning

The following is an example of computing the **average precision** in _PyTorch Lightning_:

```python

import pytorch_lightning as pl
from alm.metrics.object_detection import get_true_positives, precision_recall_from_tp, \
   average_precision_from_pr

class ObjectDetector(pl.LightningModule):

   def forward(self):
      pass # ...define model here

   def validation_step(self, sample):
      # Data sample
      gt_bbox = sample["bbox"]  # ground truth boxes in shape [batch size, n, 4]
      gt_mask = sample["mask"]  # bbox mask where True indicates bbox is valid [batch size, n]
      # Model prediction
      out = self(sample)
      pred_bbox = out["bbox"]   # prediction with shape [batch size, m, 4]
      pred_conf = out["conf"]   # confidence of prediction [batch size, m]
      # Compute true positives
      with torch.no_grad():
         trues, positives, conf = get_true_positives(gt_bbox, pred_bbox, pred_conf, gt_mask, tensor_format="xyxy")
      return {"trues": trues.cpu(), "positives": positives.cpu(), "conf": conf.cpu()}

   def validation_epoch_end(self, outputs):
      # Join all outputs into single tensor (CPU)
      trues, positives, conf = \
         (torch.cat([out[key] for out in outputs]) for key in ("trues", "positives", "conf"))
      # Compute precision recall
      precision, recall = precision_recall_from_tp(trues, positives, conf, image_dim=0)
      ap = average_precision_from_pr(precision, recall)
      self.log("val/AP", ap) # log to Tensorboard

```

Let's take a closer look.

In the line

```python
trues, positives, conf = get_true_positives(gt_bbox, pred_bbox, pred_conf, gt_mask, tensor_format="xyxy")
```

we are computing which of the bounding boxes are _true positives_ and which are _false positives_. `trues` is a boolean tensor with shape [batch size, m] indicating if the given bounding box prediction is a true positive. `positives` is an integer tensor with shape [batch size] which contains the total number of labels in each image. `conf` is a float tensor with shape [batch size, m] that contains sorted confidences (important to have correct correspondences). Note that we are passing in `tensor_format="xyxy"` which tells the method `get_true_positives` that labels and predictions are bounding boxes in format (x1, y1, x2, y2) at the last dimension (top-left and bottom-right corner of each box). We could also pass in `tensor_format="xywha"` for rotated bounding boxes in format (center x, center y, width, height, angle).

These tensors are aggregated over all validation steps by PyTorch Lightning. We calculate the final AP in the `validation_epoch_end` hook. First we concatenate all tensors. Then we compute the precision-recall-curve using the method `precision_recall_from_tp(...)`.

```python
precision, recall = precision_recall_from_tp(trues, positives, conf, image_dim=0)
```

`image_dim=0` means that we want to aggregate the precision and recall over the batch dimension (if we do not specify this, we get the precision and recall for each image seperately). The output is `precision` and `recall` which are float tensors in the range (0.0, 1.0] with shape [batch size * m].

Finally we compute the average precision using _exact integration_ of the PR-curve using the method `average_precision_from_pr(precision, recall)`. This gives us a scalar value of the Average Precision. Voilà.
