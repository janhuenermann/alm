import os
from typing import List
import torch
from torch import Tensor


def convex_convex_intersection(poly1, poly2, pad_value: float = float('inf')):
    """
    Returns the intersection of two convex polygon.
    Both polygons **must** be oriented counter-clockwise
    in south-east positive coordinate system, e.g.
    [(-1, -1), (-1, 1), (1, 1), (1, -1)] is ccw.
 
    poly1: [*, n, 2]
    poly2: [*, m, 2]
    """
    poly1, poly2 = torch.broadcast_tensors(poly1, poly2)
    if poly1.is_cuda:
        poly1, poly2 = poly1.contiguous(), poly2.contiguous()
    return torch.ops.alm_ext.sutherland_hodgman(poly1, poly2, pad_value)


def area_of_intersection(poly1, poly2, pad_value: float = float('inf')):
    """
    Returns the area of intersection of two polygons.
    Both polygons **must** be oriented counter-clockwise
    in south-east positive coordinate system, e.g.
    [(-1, -1), (-1, 1), (1, 1), (1, -1)] is ccw.

    poly1: [*, n, 2]
    poly2: [*, m, 2]
    """
    poly1, poly2 = torch.broadcast_tensors(poly1, poly2)
    if poly1.is_cuda:
        poly1, poly2 = poly1.contiguous(), poly2.contiguous()
    return torch.ops.alm_ext.compute_intersection_area(poly1, poly2, pad_value)


def convex_hull(points, pad_value: float = float('inf'), return_mask: bool = False, return_indices: bool = False) -> List[Tensor]:
    """
    Returns the convex hull of the given points.
    Empty points are replaced with pad_value.
    
    points: Tensor of shape [*, n, 2]
    """
    if points.is_cuda:
        points = points.contiguous()
    indices = torch.ops.alm_ext.convex_hull(points)
    if return_indices:
        return [indices]
    # Add dust bin
    ch = torch.cat((points.new_tensor(pad_value).expand(points.shape[:-2] + (1, 2,)), points), -2)
    # Gather from indices
    ch = ch.gather(-2, (indices[..., None] + 1).expand(indices.shape + (2,)))
    if return_mask:
        return [ch, indices >= 0]
    return [ch]


def normalize_polygon(poly, cw=False):
    """
    Normalizes polygon to be oriented clockwise in a coordinate system
    where south and east are positive (north and west negative).
    
    poly: Polygon with shape [*, 2]
    cw: True to normalize to clockwise, otherwise will normalize to ccw
    """
    p0, p1 = poly, torch.roll(poly, 1, -2)
    orient = torch.sum((p1[...,0] - p0[...,0]) * (p1[...,1] + p0[...,1]), -1)
    needs_flip = orient < 0. if cw else orient > 0.
    out = poly.clone()
    out[needs_flip] = torch.roll(poly[needs_flip].flip(-2), 1, -2)
    return out


def min_rotated_rect(convex_poly, edge=None):
    """
    Returns the smallest rotated rectangle that covers the given polygon.
    Input polygon *must* be convex.
    Output format is corner points of rectangle.
    
    poly: Convex polygon in shape [*, n, 2]
    edge: Specify an edge used for projection in shape [*, 2]
    """
    if edge is None:
        edges = convex_poly - torch.roll(convex_poly, 1, -2) # [*, n, 2]
    else:
        edges = edge[..., None, :]
    edges = edges / edges.norm(2, -1, True)
    normals = torch.stack((-edges[..., 1], edges[..., 0]), -1) # [*, n, 2]
    basis = torch.stack((edges, normals), -2) # [*, n, 2, 2]
    ps = torch.matmul(convex_poly[..., None, :, :], basis.transpose(-1, -2))  # [*, n, n, 2]
    ps0, ps1 = ps.amin(-2), ps.amax(-2)
    areas = (ps1 - ps0).prod(-1)
    k = areas.argmin(-1)[..., None, None].expand(areas.shape[:-1] + (1, 2,))
    p00 = ps0.gather(-2, k).squeeze(-2)
    p11 = ps1.gather(-2, k).squeeze(-2)
    b = basis.gather(-3, k[..., None].expand(k.shape + (2,))).squeeze(-3)
    rect = torch.stack((
        p00[..., 0], p00[..., 1],
        p00[..., 0], p11[..., 1],
        p11[..., 0], p11[..., 1],
        p11[..., 0], p00[..., 1],), -1).view(k.shape[:-2] + (4, 2,))
    rect = torch.matmul(rect, b)
    return normalize_polygon(rect)


@torch.jit.script
def shoelace(poly):
    """
    Returns the area of the polygon using the shoelace formula
    
    poly: Polygon with the shape [*, 2]
    """
    rpoly = poly.roll(-1, -2)
    s1 = (poly[...,0]*rpoly[...,1]).sum(-1, False)
    s2 = (poly[...,1]*rpoly[...,0]).sum(-1, False)
    return (s1 - s2).abs() / 2.
