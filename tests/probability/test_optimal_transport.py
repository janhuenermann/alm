import unittest
import torch
from alm.probability.optimal_transport import log_sinkhorn


class TestSinkhorn(unittest.TestCase):

    def test_sinkhorn_random(self):
        M = torch.randn((2, 3, 8, 8))
        mu = torch.ones((2, 3, 8))
        nu = mu.clone()

        log_T = log_sinkhorn(M, mu, nu, 32)
        T = log_T.exp()

        self.assertTrue(torch.allclose(T.sum(-1), torch.ones((2, 3, 8))))
        self.assertTrue(torch.allclose(T.sum(-2), torch.ones((2, 3, 8))))

    def test_sinkhorn_fixed(self):
        M = torch.tensor(
            [[0., -10., 5., 1.],
             [-1., 1., 10., 0.],
             [-3., -5., 4., 4.],
             [-2.,  2., 1., 0.]]
        )

        mu = torch.ones((4,))
        nu = mu.clone()

        log_T = log_sinkhorn(.1 * M, mu, nu, 32)
        T = log_T.exp()

        expected = torch.tensor(
            [[0.325362563, 0.130567580, 0.275252342, 0.268817514],
             [0.212763458, 0.283477187, 0.327972323, 0.175787106],
             [0.225639492, 0.201520130, 0.233151108, 0.339689255],
             [0.236234412, 0.384435177, 0.163624257, 0.215706185]])

        self.assertTrue(torch.allclose(T, expected, ))


if __name__ == '__main__':
    unittest.main()

