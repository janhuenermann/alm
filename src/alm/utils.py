import torch
from torch import Tensor
from typing import List, Optional


def compare_shape(actual_shape: List[int], target_shape: List[Optional[int]]):
    """
    The method compares two shapes. If they match, it returns True.
    Target shape can contain None to indicate arbitrary dimensions,
    or -1 for any size.

    Parameters
    ----------
    actual_shape
        List of integers, shape of tensor
    target_shape
        Target shape pattern, None for arbitary dimensions, -1 for any size

    Returns
    -------
    success
        True if matches
    """
    Q = [(0, 0)]
    while len(Q) > 0:
        i, j = Q.pop()
        if i == len(target_shape):
            if j == len(actual_shape):
                return True
            continue
        s = target_shape[i]
        if s is None:
            if i == len(target_shape) - 1:
                return True
            for j in range(len(actual_shape), j - 1, -1):
                Q.append((i + 1, j,))
        else:
            if j == len(actual_shape):
                continue
            elif s == -1 or s == actual_shape[j]:
                Q.append((i + 1, j + 1,))
    return False
 

def check_shape(tensor: Tensor, target_shape: List[Optional[int]], error_label: Optional[str] = None):
    """
    This method verifies the shape of a given tensor. If error_label is given,
    and the tensor does not match the pattern, this method raises an exception.
    Look at compare_shape for pattern.

    Parameters
    ----------
    tensor
        Tensor to be matched
    shape
        Target shape
    error_label:
        Optional string. If set, raises an exception if tensor does not match pattern shape

    Returns
    -------
    success
        True if the shape matched, False otherwise
    message
        Error message if not successful
    """
    if compare_shape(tensor.shape, target_shape):
        return True
    if error_label is not None:
        message = "Invalid input ´{}´: tensor shape ({}) does not match expected shape ({})"\
            .format(error_label, " ".join([str(x) for x in tensor.shape]), " ".join([str(x) for x in target_shape]))
        raise RuntimeError(message)
    return False


def measure_timing(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)
