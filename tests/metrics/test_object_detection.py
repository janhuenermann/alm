import unittest
import torch
from alm.metrics.object_detection import iou, xywha_to_xy4, xywh_to_xy4, box_iou, generalized_box_iou, average_precision
from alm.geometry.polygon import area_of_intersection, normalize_polygon
from math import pi, sqrt, cos, sin, sqrt

class TestBoxIOU(unittest.TestCase):

   def test_iou(self):
      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([0.5, 0.5, 1., 1.])
      result = box_iou(boxes1, boxes2)

      self.assertEqual(result.shape, tuple())
      self.assertEqual(result.item(), 0.25)

      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([
         [1.0, 1.0, 2., 2.],
         [-.1, -.1, 0., 0.],
         [-1.0, -1.0, 2., 2.],
         [0., 0., 1., 1.]
      ])
      result = box_iou(boxes1[None, :], boxes2)

      expected = [0., 0., 1./9., 1.]
      for result_i, expected_i in zip(result, expected):
         self.assertTrue(torch.isclose(result_i, torch.tensor(expected_i)), (result_i.item(), expected_i))

   def test_giou(self):
      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([0.5, 0.5, 1., 1.])
      result = generalized_box_iou(boxes1, boxes2)

      self.assertEqual(result.shape, tuple())
      self.assertEqual(result.item(), 0.25)

      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([
         [1.0, 1.0, 2., 2.],
         [-.1, -.1, 0., 0.],
         [-1.0, -1.0, 2., 2.],
         [0., 0., 1., 1.]
      ])
      result = generalized_box_iou(boxes1[None, :], boxes2)

      expected = [-0.5, -1/6.05, 1./9., 1.]
      for result_i, expected_i in zip(result, expected):
         self.assertTrue(torch.isclose(result_i, torch.tensor(expected_i)), (result_i.item(), expected_i))


class TestCoordTransform(unittest.TestCase):

   def test_xywh(self):
      out = xywh_to_xy4(torch.tensor([
         [1., 1., 0.5, 0.5],
         [0., 0., 5.0, 5.0]
      ]))

      self.assertTrue(torch.allclose(out, torch.tensor([
         [[0.75, 0.75], [0.75, 1.25], [1.25, 1.25], [1.25, 0.75]],
         [[-2.5, -2.5], [-2.5, 2.5], [2.5, 2.5], [2.5, -2.5]],
      ])), out)

   def test_xywha(self):
      xywha = torch.tensor([
         [1., 1., 0.5, 0.5, 0.0],
         [1., 1., 1.0, 2.0, pi / 180 * 30],
         [1., 3., 4.0, 2.0, pi / 180 * 80],
         [6., 3., sqrt(2) * 3, sqrt(2) * 2, pi / 180 * 10],
      ])

      for i, out in enumerate(xywha_to_xy4(xywha)):
         x, y, w, h, a = xywha[i]
         r00, r01 = w / 2 * cos(a), -h / 2 * sin(a)
         r10, r11 = w / 2 * sin(a),  h / 2 * cos(a)
         p10 = [x - r00 - r01, y - r10 - r11]
         p11 = [x - r00 + r01, y - r10 + r11]
         p12 = [x + r00 + r01, y + r10 + r11]
         p13 = [x + r00 - r01, y + r10 - r11]
         rec = torch.tensor([p10, p11, p12, p13])
         self.assertTrue(torch.allclose(out, rec), (rec, out))

   def test_xywha2(self):
      xywha = torch.tensor([
         [1., 1., 0.5, 0.5, 0.0],
         [0., 0., 5.0, 5.0, 0.0],
         [0., 0., 5.0, 5.0, -pi / 2.],
         [0., 0., 5.0, 5.0, pi / 2.],
         [0., 0., 5.0, 5.0, 5. * pi / 2.],
      ])

      out = xywha_to_xy4(xywha, upper_left_first=True)
      self.assertTrue(torch.allclose(out, torch.tensor([
         [[0.75, 0.75], [0.75, 1.25], [1.25, 1.25], [1.25, 0.75]],
         [[-2.5, -2.5], [-2.5, 2.5], [2.5, 2.5], [2.5, -2.5]],
         [[-2.5, -2.5], [-2.5, 2.5], [2.5, 2.5], [2.5, -2.5]],
         [[-2.5, -2.5], [-2.5, 2.5], [2.5, 2.5], [2.5, -2.5]],
         [[-2.5, -2.5], [-2.5, 2.5], [2.5, 2.5], [2.5, -2.5]],
      ])), out)


class TestRotatedIOU(unittest.TestCase):

   def test_iou(self):
      boxes1 = torch.tensor([0., 0., 2., 1., 0.])
      boxes2 = torch.tensor([0., 0., 2., 1., pi / 2.])
      boxes3 = torch.tensor([0., 0., 1., 1., pi])
      boxes4 = torch.tensor([0., 0., 1., 1., -pi])
      boxes5 = torch.tensor([0., 0., 1., 1., pi / 4])

      result = iou(boxes1, boxes1, "xywha")
      self.assertTrue(torch.allclose(result, torch.tensor(1.)), (result,))

      result = iou(boxes2, boxes2, "xywha")
      self.assertTrue(torch.allclose(result, torch.tensor(1.)), (result,))

      result = iou(boxes1, boxes2, "xywha")
      self.assertTrue(torch.allclose(result, torch.tensor(1. / 3.)), (result,))

      result = iou(boxes1, boxes3, "xywha")
      self.assertTrue(torch.allclose(result, torch.tensor(0.5)), (result,))

      result = iou(boxes3, boxes4, "xywha")
      self.assertTrue(torch.allclose(result, torch.tensor(1.)), (result,))

      result = iou(boxes3, boxes5, "xywha")
      self.assertTrue(torch.allclose(result, torch.tensor(1. / sqrt(2))), (result,))

   def test_stability(self):
      boxes1 = torch.tensor([0., 0., 2., 1., 0.])
      boxes2 = torch.tensor([0., 0., 2., 1., 1e-5])

      result = iou(boxes1, boxes2, "xywha")
      self.assertTrue(torch.allclose(result, torch.tensor(1.), 1e-4), (result,))

class TestAP(unittest.TestCase):

   def setUp(self):
      self.prediction = [
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
         ],
         [
            [544, 419, 621, 476, 0.4312],
            [437, 246, 518, 351, 0.5643],
            [407, 386, 531, 476, 0.1352],
            [609, 297, 636, 392, 0.9999],
            [439, 157, 556, 241, 0.4312],
            [515, 306, 595, 375, 0.2421],
         ]
      ]

      self.ground_truth = [
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
         ],
         [
            [609, 297, 636, 392],
            [437, 246, 518, 351],
            [515, 306, 595, 375],
            [439, 157, 556, 241],
            [544, 419, 621, 476],
            [407, 386, 531, 476],
         ],
      ]

      # Computed using tool from https://github.com/Cartucho/mAP
      self.ap = [0.6190, 0.5, 0.5694, 0.25, 1.0]
      self.ap_overall = 0.5866

   def test_average_precision_single(self):
      for i in range(len(self.ground_truth)):
         gt_i = torch.tensor(self.ground_truth[i])
         mask_i = torch.ones_like(gt_i[:, 0], dtype=torch.bool)
         pred_i = torch.tensor(self.prediction[i])

         ap_i = average_precision(gt_i, pred_i[:, :4], pred_i[:, 4], mask_i)
         self.assertTrue(torch.isclose(ap_i, torch.tensor(self.ap[i]), atol=1e-4), (ap_i, self.ap[i]))

   def test_average_precision_batched(self):
      gt_max_len = max(len(gt_i) for gt_i in self.ground_truth)
      pred_max_len = max(len(pred_i) for pred_i in self.prediction)

      gt_batch = []
      gt_mask = []
      pred_batch = []
      pred_mask = []

      for gt_i in self.ground_truth:
         gt_pad_i = gt_max_len - len(gt_i)
         gt_batch.append(gt_i + [[0,0,0,0]] * gt_pad_i)
         gt_mask.append(len(gt_i) * [True] + gt_pad_i * [False])

      for pred_i in self.prediction:
         pred_pad_i = pred_max_len - len(pred_i)
         pred_batch.append(pred_i + [[0,0,0,0,0]] * pred_pad_i)
         pred_mask.append(len(pred_i) * [True] + pred_pad_i * [False])

      gt_batch = torch.tensor(gt_batch, dtype=torch.float)
      gt_mask = torch.tensor(gt_mask, dtype=torch.bool)
      pred_batch = torch.tensor(pred_batch, dtype=torch.float)
      pred_mask = torch.tensor(pred_mask, dtype=torch.bool)

      ap = average_precision(gt_batch, pred_batch[..., :4], pred_batch[..., 4], gt_mask, pred_mask)
      self.assertTrue(torch.allclose(ap, torch.tensor(self.ap), atol=1e-4), (ap, self.ap))

      overall_ap = average_precision(gt_batch, pred_batch[..., :4], pred_batch[..., 4], gt_mask, pred_mask, image_dim=-2)
      self.assertTrue(torch.allclose(overall_ap, torch.tensor(self.ap_overall), atol=1e-4), (overall_ap, self.ap_overall))


if __name__ == '__main__':
    unittest.main()

