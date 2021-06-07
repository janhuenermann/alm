import unittest
import torch
from alm.geometry.polygon import shoelace, min_rotated_rect, normalize_polygon, convex_convex_intersection,\
    convex_hull, area_of_intersection


rectangle = torch.tensor([[ -1., -1. ], [ -1., 1. ], [ 1., 1. ], [ 1., -1. ]])
convex_polygon = torch.tensor([[0., 1.], [0., 2.], [4., 6.], [6., 3.], [4., 0.], [2., 0.]])
concave_polygon = torch.tensor([[0., 1.], [0., 2.], [2., 2.], [4., 6.], [6., 3.], [4., 0.], [2., 0.]])


class TestPolygonBasics(unittest.TestCase):

    def test_shoelace(self):
        self.assertEqual(shoelace(rectangle), 4.0)
        self.assertEqual(shoelace(rectangle.flip(-2)), 4.0)
        self.assertEqual(shoelace(concave_polygon), 17.0)

    def test_normalize_cw(self):
        cw = torch.roll(rectangle.flip(-2), 1, 0)
        self.assertTrue(torch.allclose(normalize_polygon(rectangle), rectangle))
        self.assertTrue(torch.allclose(normalize_polygon(cw), rectangle))


class TestRotatedRect(unittest.TestCase):

    def test_min_rotated_rect(self):
        rect = min_rotated_rect(convex_polygon)
        expected = torch.tensor(
            [[ 2.5, -1.5],
             [-0.5,  1.5],
             [ 4. ,  6. ],
             [ 7. ,  3. ]])
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
        ch, mask = convex_hull(concave_polygon, return_mask=True)
        match = [torch.allclose(convex_polygon.roll(i, -2), ch[mask]) for i in range(len(convex_polygon))]
        self.assertTrue(any(match), (concave_polygon, convex_polygon))

    def test_batching(self):
        ch = convex_hull(rectangle.expand(100, -1, -1))[0]
        self.assertTrue(torch.allclose(ch[0], rectangle), (ch, rectangle))

    def test_duplicates(self):
        # Repeat
        points = concave_polygon.expand(10, -1, -1).reshape(-1, 2)
        ch, mask = convex_hull(points, return_mask=True)
        match = [torch.allclose(convex_polygon.roll(i, -2), ch[mask]) for i in range(len(convex_polygon))]
        self.assertTrue(any(match), (points, ch))

        # Repeat interleave
        points = concave_polygon[:, None].expand(-1, 10, -1).reshape(-1, 2)
        ch, mask = convex_hull(points, return_mask=True)
        match = [torch.allclose(convex_polygon.roll(i, -2), ch[mask]) for i in range(len(convex_polygon))]
        self.assertTrue(any(match), (points, ch))

    def test_simple(self):
        ch = convex_hull(rectangle)[0]
        self.assertTrue(torch.allclose(ch, rectangle), (ch, rectangle))

        rect_with_interior = torch.cat((rectangle, 2. * torch.rand((100, 2,)) - 1.), -2)
        ch, mask = convex_hull(rect_with_interior, return_mask=True)
        self.assertTrue(torch.allclose(ch[mask], rectangle), (ch, rectangle))

    def test_zeros(self):
        points = torch.zeros((16, 2))
        ch, mask = convex_hull(points, return_mask=True)
        self.assertTrue(torch.allclose(ch[mask], torch.tensor([[0., 0.]])))

    def test_colinear(self):
        points = torch.tensor([[0., 0.], [-5., 0.], [10., 0.], [8., 0.], [1., 1.]])
        ch, mask = convex_hull(points, return_mask=True)
        self.assertTrue(mask.sum() == 3)
        self.assertTrue(torch.allclose(ch[mask], torch.tensor([[-5., 0.], [1., 1.], [10., 0.]])))


class TestConvexConvexIntersection(unittest.TestCase):

    def test_simple(self):
        inter = convex_convex_intersection(rectangle, rectangle)
        self.assertTrue(torch.allclose(inter[:4], rectangle))

        inter = convex_convex_intersection(rectangle, rectangle + 1.)
        self.assertTrue(torch.allclose(inter[:4], torch.tensor([
            [0., 0.],
            [0., 1.],
            [1., 1.],
            [1., 0.]
        ])), (inter,))


    def test_intersection_area(self):
        A = torch.tensor([
                [-1.0000, -0.5000],
                [-1.0000,  0.5000],
                [ 1.0000,  0.5000],
                [ 1.0000, -0.5000]])
        B = torch.tensor([
                [-0.5000,  1.0000],
                [ 0.5000,  1.0000],
                [ 0.5000, -1.0000],
                [-0.5000, -1.0000]])
        area = area_of_intersection(A, B)
        self.assertTrue(torch.allclose(area, torch.tensor(1.)), (area,))


if __name__ == '__main__':
    unittest.main()

