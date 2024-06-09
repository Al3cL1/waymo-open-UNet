import tensorflow as tf
import numpy as np
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils.lidar_utils import convert_range_image_to_point_cloud


def convert_range_image_to_point_cloud_labels(
    range_image: v2.perception.lidar.RangeImage,
    segmentation_label: v2.perception.segmentation.LiDARSegmentationRangeImage
) -> np.ndarray:
    """Converts a lidar frame's labels into a point cloud label representation.

    Args:
        range_image: the lidar RangeImage.
        segmentation_label: the lidar's respective LiDARSegmentationRangeImage.

    Returns:
        np.ndarray: [N, 2] numpy array, corresponding to {instance_id, semantic_class}.
    """
    range_image_tensor = range_image.tensor
    range_image_mask = range_image_tensor[..., 0] > 0
    sl_tensor = segmentation_label.tensor
    sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
    return sl_points_tensor.numpy()


def convert_lidar_range_to_point_cloud(
    range_image: v2.perception.lidar.RangeImage, 
    calibration: v2.perception.context.LiDARCalibrationComponent,
    lidar_pose_range_image: v2.perception.lidar.RangeImage,
    frame_pose: v2.perception.pose.VehiclePoseComponent
) -> np.ndarray:
    """Converts a lidar frame's range image into a point cloud representation.

    Args:
        lidar: the lidar Range_Image.
        calibration: the lidar RangeImage's respective calibration component.
        lidar_pose: the lidar RangeImage's respective pose range image.
        frame_pose: the lidar's vehicle pose component.

    Returns:
        np.ndarray: A [N, 3] tensor of 3D LiDAR points, with 3 corresponding to (x, y, z).
    """
    points_tensor = convert_range_image_to_point_cloud(
        range_image=range_image,
        calibration=calibration,
        pixel_pose=lidar_pose_range_image,
        frame_pose=frame_pose,
        keep_polar_features=False
    )
    points = points_tensor.numpy()
    return points[:, :3]