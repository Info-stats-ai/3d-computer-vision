"""3D IoU between oriented bounding boxes (KITTI-style: BEV polygon IoU x height overlap)."""
import numpy as np

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover
    Polygon = None


def box_corners_bev(centroid, length, width, yaw):
    """Bird's-eye-view (x-z plane in camera coords) corners of a rotated box."""
    x, _, z = centroid
    dx, dz = length / 2.0, width / 2.0
    corners = np.array([[dx, dz], [dx, -dz], [-dx, -dz], [-dx, dz]])
    cos, sin = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = corners @ rot.T
    corners[:, 0] += x
    corners[:, 1] += z
    return corners


def iou_3d(box_a, box_b):
    """box_a, box_b: dict with 'centroid' [x,y,z], 'dimensions' {length,width,height}, 'yaw'.
    Returns scalar 3D IoU using BEV polygon intersection x vertical overlap,
    matching the standard KITTI evaluation protocol."""
    if Polygon is None:
        raise ImportError("shapely is required for 3D IoU computation (pip install shapely)")

    ca, cb = box_a["centroid"], box_b["centroid"]
    da, db = box_a["dimensions"], box_b["dimensions"]

    poly_a = Polygon(box_corners_bev(ca, da["length"], da["width"], box_a["yaw"]))
    poly_b = Polygon(box_corners_bev(cb, db["length"], db["width"], box_b["yaw"]))
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0

    inter_area = poly_a.intersection(poly_b).area
    if inter_area <= 0:
        return 0.0

    # Vertical (height) overlap. In camera coords y points down, so the box
    # spans from (y - height) up to y.
    ya_top, ya_bot = ca[1] - da["height"], ca[1]
    yb_top, yb_bot = cb[1] - db["height"], cb[1]
    y_overlap = max(0.0, min(ya_bot, yb_bot) - max(ya_top, yb_top))
    if y_overlap <= 0:
        return 0.0

    inter_vol = inter_area * y_overlap
    vol_a = poly_a.area * da["height"]
    vol_b = poly_b.area * db["height"]
    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol
