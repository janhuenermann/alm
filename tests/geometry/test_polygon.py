import unittest
import numpy as np
from geometry.polygon import shoelace, min_rotated_rect, normalize_polygon


rectangle = np.array([[ -1., -1. ], [ 1., -1. ], [ 1., 1. ], [ -1., 1. ]])
concave_polygon = np.array([[0., 1.], [0., 2.], [2., 2.], [4., 6.], [6., 3.], [4., 0.], [2., 0.]])


class TestPolygon(unittest.TestCase):

    def test_shoelace(self):
        self.assertEqual(shoelace(rectangle), 4.0)
        self.assertEqual(shoelace(rectangle[::-1]), 4.0)
        # Repeat interleave
        self.assertEqual(shoelace(np.repeat(rectangle, 5, 0)), 4.0)
        # Repeat
        self.assertEqual(shoelace(np.tile(rectangle, (5, 1)), strict=True), 4.0)

    def test_normalize_cw(self):
        np.testing.assert_almost_equal(
            normalize_polygon(rectangle), rectangle)
        np.testing.assert_almost_equal(
            normalize_polygon(np.roll(rectangle[::-1], 1, 0)), rectangle)

    def test_min_rotated_rect(self):
        rect = min_rotated_rect(concave_polygon)
        expected = np.array(
            [[ 4. ,  6. ],
             [-0.5,  1.5],
             [ 2.5, -1.5],
             [ 7. ,  3. ]])
        np.testing.assert_almost_equal(rect, expected, 4)


if __name__ == '__main__':
    unittest.main()

