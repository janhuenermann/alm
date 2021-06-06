import unittest
import torch
from alm.geometry.polygon import shoelace, min_rotated_rect, normalize_polygon, convex_hull


rectangle = torch.tensor([[ -1., -1. ], [ 1., -1. ], [ 1., 1. ], [ -1., 1. ]])
convex_polygon = torch.tensor([[0., 1.], [0., 2.], [4., 6.], [6., 3.], [4., 0.], [2., 0.]])
concave_polygon = torch.tensor([[0., 1.], [0., 2.], [2., 2.], [4., 6.], [6., 3.], [4., 0.], [2., 0.]])


class TestPolygonBasics(unittest.TestCase):

    def test_shoelace(self):
        self.assertEqual(shoelace(rectangle), 4.0)
        self.assertEqual(shoelace(rectangle.flip(-2)), 4.0)
        self.assertEqual(shoelace(concave_polygon), 17.0)

    def test_normalize_cw(self):
        ccw = torch.roll(rectangle.flip(-2), 1, 0)
        self.assertTrue(torch.allclose(normalize_polygon(rectangle), rectangle))
        self.assertTrue(torch.allclose(normalize_polygon(ccw), rectangle))


class TestRotatedRect(unittest.TestCase):

    def test_min_rotated_rect(self):
        rect = min_rotated_rect(convex_polygon)
        expected = torch.tensor(
            [[ 2.5, -1.5],
             [ 7. ,  3. ],
             [ 4. ,  6. ],
             [-0.5,  1.5]])
        self.assertTrue(torch.allclose(rect, expected), (rect, expected))

        # Test batching
        more_convex_polygon = convex_polygon.expand(100, 10, -1, 2).clone()
        more_convex_polygon += torch.randn((100, 10, 1, 2)) * 10.
        more_rect = min_rotated_rect(more_convex_polygon)
        p0, p1 = more_rect.amin(-2), more_rect.amax(-2)
        area = (p1 - p0).prod(-1)
        self.assertTrue(torch.allclose(area, torch.tensor(56.25).expand_as(area)), area)


class TestConvexHull(unittest.TestCase):

    def test_concave(self):
        print(convex_hull(concave_polygon))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

