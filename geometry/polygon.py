import numpy as np
from cv2 import convexHull


def normalize_polygon(poly, cw=True):
    """
    Returns a polygon which is always oriented clockwise
    
    poly: Polygon with shape [*, 2]
    cw: True to normalize to clockwise, otherwise will normalize to ccw
    """
    p0, p1 = poly, np.roll(poly, 1, 0)
    is_ccw = np.sum((p1[:, 0] - p0[:, 0]) * (p1[:, 1] + p0[:, 1])) < 0.
    if is_ccw == cw:
        return np.roll(poly[::-1], 1, 0)
    return poly


def min_rotated_rect(poly):
    """
    Returns the four points of the smallest rotated rect that covers the given polygon
    
    poly: Polygon with shape [*, 2]
    """
    ch = convexHull(poly.astype(np.float32))[:, 0]
    edges = ch - np.roll(ch, 1, 0)
    edges = edges / np.linalg.norm(edges, 2, -1)[:, None]
    normals = np.stack((-edges[:, 1], edges[:, 0]), 1)
    basis = np.stack((edges, normals), -2)
    ps = np.matmul(ch, basis.swapaxes(1, 2))
    ps0 = np.amin(ps, 1)
    ps1 = np.amax(ps, 1)
    areas = np.prod(ps1 - ps0, -1)
    k = np.argmin(areas)
    rect = np.array(
        [[ps0[k, 0], ps0[k, 1]],
         [ps0[k, 0], ps1[k, 1]],
         [ps1[k, 0], ps1[k, 1]],
         [ps1[k, 0], ps0[k, 1]]])
    return normalize_polygon(rect @ basis[k])


def shoelace(poly, strict=False):
    """
    Returns the area of the polygon using the shoelace formula
    
    poly: Polygon with the shape [*, 2]
    strict: Whether to remove duplicate points from the polygon
    """
    if strict:
        _, order = np.unique(poly, axis=0, return_index=True)
        poly = poly[np.sort(order)]
    s1 = np.sum(poly[:,0] * np.roll(poly[:,1], -1))
    s2 = np.sum(poly[:,1] * np.roll(poly[:,0], -1))
    return np.absolute(s1 - s2) / 2.
