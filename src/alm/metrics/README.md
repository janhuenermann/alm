# `metrics`

## Object Detection

Example of computing average precision in _PyTorch Lightning_:

```python

import pytorch_lightning as pl
from alm.metrics.object_detection import get_true_positives, precision_recall_from_tp, \
   average_precision_from_pr

class ObjectDetector(pl.LightningModule):

   def forward(self):
      # ...define model here
      pass

   def validation_step(self, sample):
      # Sample contains:
      # - "image" ....
      # - "bbox" key: ground truth boxes in shape [batch size, n, 4]
      # - "mask" key: bbox mask where True indicates bbox is valid [batch size, n]
      gt_bbox = sample["bbox"]
      gt_mask = sample["mask"]

      # Model returns dict with
      # - "bbox" key: prediction with shape [batch size, m, 4]
      # - "conf" key: confidence of prediction [batch size, m]
      out = self(sample)

      pred_bbox = out["bbox"]
      pred_conf = out["conf"]

      with torch.no_grad():
         # Take a look at `object_detection.py` for more information on `get_true_positives`
         trues, positives, conf = get_true_positives(gt_bbox, pred_bbox, pred_conf, gt_mask)
         trues, positives, conf = (x.cpu() for x in (trues, positives, conf))

      return {"trues": trues, "positives": positives, "conf": conf}

   def validation_epoch_end(self, outputs):
      # Join all outputs into single tensor (CPU)
      trues, positives, conf = \
         (torch.cat([out[key] for out in outputs]) for key in ("trues", "positives", "conf"))
      # Compute precision recall
      precision, recall = precision_recall_from_tp(trues, positives, conf, image_dim=0)
      # Get AP
      ap = average_precision_from_pr(precision, recall)
      # Log
      self.log("val/AP", ap)

```