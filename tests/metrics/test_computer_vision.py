import unittest
import torch
from metrics.computer_vision import iou, generalized_iou

class TestIOU(unittest.TestCase):

   def test_iou(self):
      boxes1 = torch.tensor([0., 0., 1., 1.])
      boxes2 = torch.tensor([0.5, 0.5, 1., 1.])
      result = iou(boxes1, boxes2)

      self.assertEqual(result.shape, tuple())
      self.assertTrue(result.item() == 0.25)

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
      self.assertTrue(result.item() == 0.25)

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

if __name__ == '__main__':
    unittest.main()

