import torch
from torch import Tensor
from typing import List, Optional


def equal_shape(tensor: Tensor, shape: List[Optional[int]], error_label: Optional[str] = None):
    """
    This method verifies the shape of a given tensor. The given shape
    can have None to indicate arbitrary dimensions in between.

    Parameters
    ----------
    tensor
        Tensor to be matched
    shape
        Target shape

    Returns
    -------
    success
        True if the shape matched, False otherwise
    message
        Error message if not successful
    """
    Q = [(0, 0)]
    while len(Q) > 0:
        i, j = Q.pop()
        if i == len(shape) and j == tensor.dim():
            return True
        elif i >= len(shape):
            continue
        s = shape[i]
        if s is None:
            if i == len(shape) - 1:
                return True
            for j in range(tensor.dim(), j - 1, -1):
                Q.append((i + 1, j,))
        else:
            if j >= tensor.dim():
                continue
            elif s == -1 or s == tensor.shape[j]:
                Q.append((i + 1, j + 1,))
    if error_label is not None:
        message = "Invalid input ´{}´: tensor shape ({}) does not match expected shape ({})"\
            .format(error_label, " ".join([str(x) for x in tensor.shape]), " ".join([str(x) for x in shape]))
        raise RuntimeError(message)
    return False


def measure_timing(fn):
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)
