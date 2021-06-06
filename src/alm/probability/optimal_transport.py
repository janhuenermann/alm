import torch
from torch import Tensor


@torch.jit.script
def log_sinkhorn(score: Tensor, mu: Tensor, nu: Tensor, iters: int):
    """
    Computes the entropy-regularized optimal transport distances
    using the Sinkhorn algorithm in log-space.
    
    score: Score matrix with shape [*, m, n]
    mu: Row marginals with shape [*, m]
    nu: Column marginals with shape [*, n]
    iters: Iterations to run
    """
    m, n = score.shape[-2:]
    log_mu, log_nu = mu.log(), nu.log()
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    mask_mu, mask_nu = ( log_mu == float('-inf') ), ( log_nu == float('-inf') )
    score[mask_mu[..., :, None] & mask_nu[..., None, :]] = float('-inf')
    for _ in range(iters):
        v = log_nu - torch.logsumexp(score + u.unsqueeze(-1), dim=-2)
        v[mask_nu] = float('-inf')
        u = log_mu - torch.logsumexp(score + v.unsqueeze(-2), dim=-1)
        u[mask_mu] = float('-inf')
    return score + u.unsqueeze(-1) + v.unsqueeze(-2)
