"""Point cloud preprocessing: voxel downsampling and ground plane removal."""
import numpy as np
import open3d as o3d


def to_o3d_pointcloud(points_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz[:, :3])
    return pcd


def voxel_downsample(points_xyz, voxel_size=0.1):
    """Merge points within each voxel_size-meter grid cell into one point.
    Cuts a ~100k-point raw scan down to a tractable size while preserving
    overall shape. Returns [M, 3] downsampled points."""
    pcd = to_o3d_pointcloud(points_xyz)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down.points)


def remove_ground_plane(points_xyz, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    """Fit the dominant plane (the road surface, in an outdoor LiDAR scan) via
    RANSAC and split points into (non_ground, ground, plane_model).

    plane_model is (a, b, c, d) for the plane a*x + b*y + c*z + d = 0.
    """
    pcd = to_o3d_pointcloud(points_xyz)
    plane_model, inlier_idx = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    inlier_mask = np.zeros(len(points_xyz), dtype=bool)
    inlier_mask[inlier_idx] = True
    ground = points_xyz[inlier_mask]
    non_ground = points_xyz[~inlier_mask]
    return non_ground, ground, plane_model


def crop_range(points_xyz, x_range=(-40, 40), y_range=(-40, 40), z_range=(-3, 1)):
    """Crop a scan to a region of interest around the sensor (typical KITTI ROI)."""
    mask = (
        (points_xyz[:, 0] >= x_range[0]) & (points_xyz[:, 0] <= x_range[1]) &
        (points_xyz[:, 1] >= y_range[0]) & (points_xyz[:, 1] <= y_range[1]) &
        (points_xyz[:, 2] >= z_range[0]) & (points_xyz[:, 2] <= z_range[1])
    )
    return points_xyz[mask]
