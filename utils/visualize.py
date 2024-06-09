import numpy as np
import dask.dataframe as dd
from waymo_open_dataset import v2
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Tuple
import conversions


def read(tag: str, dataset_dir: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = f'{dataset_dir}/{tag}/{context_name}.parquet'
    return dd.read_parquet(paths)


def get_first_frame_components(
    dataset_dir: str, context_name: str
) -> Tuple[v2.perception.lidar.LiDARComponent, 
           v2.perception.segmentation.LiDARSegmentationRangeImage,
           v2.perception.context.LiDARCalibrationComponent,
           v2.perception.lidar.LiDARPoseComponent,
           v2.perception.pose.VehiclePoseComponent]:
    """Grabs lidar components from the first frame of a drive.

    Args:
        dataset_dir: the directory containing the component folders.
        context_name: the drive identifier.

    Returns:
        LiDARComponent: the drive first frame's lidar component.
        LiDARSegmentationRangeImage: the drive first frame's segmentation labels.
        LiDARCalibrationComponent: the drive first frame's lidar calibration settings.
        LiDARPoseComponent: the drive first frame's lidar pixel pose component.
        VehiclePoseComponent: the drive's vehicle pose component.
    """
    lidar_df = read('lidar', dataset_dir, context_name)
    lidar_segmentation = read('lidar_segmentation', dataset_dir, context_name)
    lidar_calibration = read('lidar_calibration', dataset_dir, context_name)
    lidar_pose = read('lidar_pose', dataset_dir, context_name)
    vehicle_pose = read('vehicle_pose', dataset_dir, context_name)

    lidar_w_seg_df = v2.merge(lidar_df, lidar_segmentation)
    lidar_w_seg_w_calib_df = v2.merge(lidar_w_seg_df, lidar_calibration)
    lidar_w_seg_w_calib_w_pose_df = v2.merge(lidar_w_seg_w_calib_df, lidar_pose)
    full_lidar_df = v2.merge(lidar_w_seg_w_calib_w_pose_df, vehicle_pose)

    _, first_row = next(iter(full_lidar_df.iterrows()))

    lidar = v2.LiDARComponent.from_dict(first_row)
    lidar_seg = v2.LiDARSegmentationLabelComponent.from_dict(first_row)
    calibration = v2.LiDARCalibrationComponent.from_dict(first_row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(first_row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(first_row)

    return (lidar, lidar_seg, calibration, lidar_pose, vehicle_pose)


def create_colormap(labels):
    """Creates a colormap based on a list of given labels for visualization.

    Args:
        labels: A [N, 1] list of semantic_class labels

    Returns:
        np.ndarray: A {semantic_class: (r, g, b)} dict relating each unique 
        semantic_class to a unique color.
    """
    colors = plt.get_cmap('tab20').colors

    unique_labels = np.unique(labels)
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    return color_map


def visualize_random_lidar_seg(dataset_dir, context_name):
    """Visualizes the segmentation data from the first frame of a given drive.

    Args:
        dataset_dir: the directory containing the component folders.
        context_name: the drive identifier.
    """
    lidar, lidar_seg, calibration, lidar_pose, frame_pose = \
    get_first_frame_components(dataset_dir, context_name)

    point_cloud = conversions.convert_lidar_range_to_point_cloud(
        lidar.range_image_return1, 
        calibration, 
        lidar_pose.range_image_return1, 
        frame_pose
    )

    labels = conversions.convert_range_image_to_point_cloud_labels(
        lidar.range_image_return1,
        lidar_seg.range_image_return1
    )

    color_map = create_colormap(np.unique(labels[..., 1]))
    point_colors = np.array([color_map[label] for label in labels[..., 1]])

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    pc.colors = o3d.utility.Vector3dVector(point_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)
    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()



dataset_dir = '../data/training'
context_name = '10072140764565668044_4060_000_4080_000'

visualize_random_lidar_seg(dataset_dir, context_name)