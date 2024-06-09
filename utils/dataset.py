import torch
import numpy as np
from torch.utils import data
import dask.dataframe as dd
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils.lidar_utils import convert_range_image_to_point_cloud
import conversions
import os
from typing import Tuple, List

TYPE_UNDEFINED = 0


def read(tag: str, dataset_dir: str, context_name: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = f'{dataset_dir}/{tag}/{context_name}'
    return dd.read_parquet(paths)


def waymo_dataset_generator(dataset_dir: str, dataset_type: str
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generates point clouds and labels from the Waymo dataset.

    Args:
        dataset_dir: the directory containing the dataset.
        dataset_type: the type of dataset (e.g., 'training', 'validation').

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing a list of xyz 
        point clouds and a list of corresponding semantic_class labels.
    """
    full_path = f"{dataset_dir}/{dataset_type}"
    try:
        file_list = os.listdir(full_path + '/lidar')
    except FileNotFoundError:
        raise FileNotFoundError(f"The lidar folder does not exist.") 
    point_clouds = []
    labels = []
    for file_name in file_list:
        lidar_df = read('lidar', full_path, file_name)
        lidar_segmentation = read('lidar_segmentation', full_path, file_name)
        lidar_calibration = read('lidar_calibration', full_path, file_name)
        lidar_pose = read('lidar_pose', full_path, file_name)
        vehicle_pose = read('vehicle_pose', full_path, file_name)

        lidar_w_seg_df = v2.merge(lidar_df, lidar_segmentation)
        lidar_w_seg_w_calib_df = v2.merge(lidar_w_seg_df, lidar_calibration)
        lidar_w_seg_w_calib_w_pose_df = v2.merge(lidar_w_seg_w_calib_df, lidar_pose)
        full_lidar_df = v2.merge(lidar_w_seg_w_calib_w_pose_df, vehicle_pose)

        for _, data_row in iter(full_lidar_df.iterrows()):
            lidar = v2.LiDARComponent.from_dict(data_row)
            lidar_seg = v2.LiDARSegmentationLabelComponent.from_dict(data_row)
            calibration = v2.LiDARCalibrationComponent.from_dict(data_row)
            lidar_pose = v2.LiDARPoseComponent.from_dict(data_row)
            vehicle_pose = v2.VehiclePoseComponent.from_dict(data_row)

            point_cloud_1 = conversions.convert_lidar_range_to_point_cloud(
                lidar.range_image_return1, 
                calibration, 
                lidar_pose.range_image_return1, 
                vehicle_pose
            )
            labels_1 = conversions.convert_range_image_to_point_cloud_labels(
                lidar.range_image_return1,
                lidar_seg.range_image_return1
            )

            point_clouds.append(point_cloud_1)
            labels.append(labels_1[:, 1])
    point_clouds = np.concatenate(point_clouds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return point_clouds, labels


class waymo_dataset(data.Dataset):
    def __init__(self, dataset_dir, dataset_type):
        self.data, self.label = waymo_dataset_generator(dataset_dir, dataset_type)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_loader = DataLoader(waymo_dataset('data', 'training'), num_workers=4,
                              batch_size=64, shuffle=True)
    i = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | lable shape: {label.shape}")
        if i == 10:
            break
        else:
            i += 1