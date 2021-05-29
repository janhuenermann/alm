import unittest
import torch
from metrics.computer_vision import iou, generalized_iou, average_precision

class TestIOU(unittest.TestCase):

   def test_iou(self):
      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([0.5, 0.5, 1., 1.])
      result = iou(boxes1, boxes2)

      self.assertEqual(result.shape, tuple())
      self.assertEqual(result.item(), 0.25)

      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([
         [1.0, 1.0, 2., 2.],
         [-.1, -.1, 0., 0.],
         [-1.0, -1.0, 2., 2.],
         [0., 0., 1., 1.]
      ])
      result = iou(boxes1[None, :], boxes2)

      expected = [0., 0., 1./9., 1.]
      for result_i, expected_i in zip(result, expected):
         self.assertTrue(torch.isclose(result_i, torch.tensor(expected_i)), (result_i.item(), expected_i))

   def test_giou(self):
      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([0.5, 0.5, 1., 1.])
      result = generalized_iou(boxes1, boxes2)

      self.assertEqual(result.shape, tuple())
      self.assertEqual(result.item(), 0.25)

      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([
         [1.0, 1.0, 2., 2.],
         [-.1, -.1, 0., 0.],
         [-1.0, -1.0, 2., 2.],
         [0., 0., 1., 1.]
      ])
      result = generalized_iou(boxes1[None, :], boxes2)

      expected = [-0.5, -1/6.05, 1./9., 1.]
      for result_i, expected_i in zip(result, expected):
         self.assertTrue(torch.isclose(result_i, torch.tensor(expected_i)), (result_i.item(), expected_i))

   def test_average_precision(self):
      pred = [
         [
            [585, 326, 639, 384, 0.271981],
            [501, 225, 610, 259, 0.473745],
            [531, 262, 584, 356, 0.356840],
            [481, 400, 570, 460, 0.354335],
            [466, 427, 585, 470, 0.354973],
            [503, 257, 552, 350, 0.352939],
            [508, 167, 625, 246, 0.325701],
         ],
         [
            [403, 384, 517, 461, 0.382881],
            [433, 272, 499, 341, 0.272826],
            [433, 260, 506, 336, 0.269833],
            [592, 310, 634, 388, 0.298196],
            [429, 219, 528, 247, 0.460851],
            [405, 429, 519, 470, 0.369369],
            [518, 314, 603, 369, 0.462608],
            [413, 390, 515, 459, 0.619459],
         ],
         [
            [63, 77, 560, 477, 0.386569],
            [92, 37, 193, 121, 0.523199],
            [292, 6, 438, 99, 0.374142],
            [3, 12, 78, 153, 0.529134],
            [13, 20, 124, 173, 0.346044],
            [436, 0, 564, 105, 0.273336],
         ],
         [
            [529, 201, 593, 309, 0.313999],
            [482, 0, 638, 275, 0.537946],
         ]
      ]

      gt = [
         [
            [477, 401, 592, 476],
            [506, 254, 599, 361],
            [514, 159, 639, 251],
            [593, 330, 637, 387]
         ],
         [
            [439, 157, 556, 241],
            [609, 297, 636, 392],
            [515, 306, 595, 375],
            [437, 246, 518, 351],
            [407, 386, 531, 476],
            [544, 419, 621, 476],
         ],
         [
            [277, 2, 444, 101],
            [93, 37, 194, 121],
            [11, 152, 84, 250],
            [469, 4, 552, 91],
            [516, 5, 638, 410],
            [4, 13, 79, 154],
         ],
         [
            [47, 115, 84, 199],
            [528, 213, 602, 300],
         ]
      ]

      # Computed using tool from https://github.com/Cartucho/mAP
      AP = [0.6190, 0.5, 0.5694, 0.25]

      for i in range(len(gt)):
         gt_i = torch.tensor(gt[i])
         mask_i = torch.ones_like(gt_i[:, 0], dtype=torch.bool)
         pred_i = torch.tensor(pred[i])

         sorted_pred_i = pred_i[torch.argsort(pred_i[:, 4], descending=True)][:, :4]
         ap_i = average_precision(gt_i, sorted_pred_i, mask_i)
         self.assertTrue(torch.isclose(ap_i, torch.tensor(AP[i]), atol=1e-4))

      gt_max_len = max(len(gt_i) for gt_i in gt)
      pred_max_len = max(len(pred_i) for pred_i in pred)

      gt_batch = []
      gt_mask = []
      pred_batch = []
      pred_mask = []

      for gt_i in gt:
         gt_pad_i = gt_max_len - len(gt_i)
         gt_batch.append(gt_i + [[0,0,0,0]] * gt_pad_i)
         gt_mask.append(len(gt_i) * [True] + gt_pad_i * [False])

      for pred_i in pred:
         pred_pad_i = pred_max_len - len(pred_i)
         pred_batch.append(pred_i + [[0,0,0,0,0]] * pred_pad_i)
         pred_mask.append(len(pred_i) * [True] + pred_pad_i * [False])

      gt_batch = torch.tensor(gt_batch, dtype=torch.float)
      gt_mask = torch.tensor(gt_mask, dtype=torch.bool)
      pred_batch = torch.tensor(pred_batch, dtype=torch.float)
      pred_mask = torch.tensor(pred_mask, dtype=torch.bool)

      sorting_batch = torch.argsort(pred_batch[..., 4], descending=True)
      sorted_pred_batch = pred_batch[..., :4].gather(-2, sorting_batch.unsqueeze(-1).expand(-1, -1, 4))
      sorted_pred_mask = pred_mask.gather(-1, sorting_batch)
      ap = average_precision(gt_batch, sorted_pred_batch, gt_mask, sorted_pred_mask)
      self.assertTrue(torch.allclose(ap, torch.tensor(AP), atol=1e-4), (ap, AP))

if __name__ == '__main__':
    unittest.main()

