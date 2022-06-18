from torch.utils.data import DataLoader
from torch.utils import data
import numpy as np
from pathlib import Path
import pickle
import cv2 as cv

import pc_processor
import torch.nn as nn

map_name_from_general_to_segmentation_class = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.vegetation': 'vegetation',
    'noise': 'ignore',
    'static.other': 'ignore',
    'vehicle.ego': 'ignore'
}

map_name_from_segmentation_class_to_segmentation_index = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'construction_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10,
    'driveable_surface': 11,
    'other_flat': 12,
    'sidewalk': 13,
    'terrain': 14,
    'manmade': 15,
    'vegetation': 16
}

class new_nuScenes(data.Dataset):
    def __init__(self, dataset_root, split):
        print(dataset_root)
        # assert Path(dataset_root).is_dir()
        self.root = dataset_root
        self.split = split
        self.train_dataset_files = list((Path(dataset_root) / 'train').glob("*.pkl"))
        self.val_dataset_files = list((Path(dataset_root) / 'val').glob("*.pkl"))
        # self.train_dataset_files = self.train_dataset_files[:250]
        # self.val_dataset_files = self.val_dataset_files[:50]

        # self.test_dataset_files = list(Path(dataset_root) / 'test'.glob("*.pkl"))
        self.mapped_cls_name = {}

        self.img_size=[512, 640]
        for v, k in map_name_from_segmentation_class_to_segmentation_index.items():
            self.mapped_cls_name[k] = v

    def __getitem__(self, index):
        assert (self.split in ['train', 'val', 'test'])
        if self.split == 'train':
            with open(self.train_dataset_files[index], 'rb') as f:
                train_data = pickle.load(f)

            points = train_data['points'].numpy().squeeze()
            labels = train_data['labels'].numpy().squeeze()
            image = train_data['camera_img'].numpy().squeeze() # only front camera

            masked_points = points[points[:, -1] == -1.0]
            masked_labels = labels[points[:, -1] == -1.0]

            masked_h = np.floor((masked_points[:, 5] + 1) * ((self.img_size[0] - 1) / 2)).astype('int64')
            masked_w = np.floor((masked_points[:, 6] + 1) * ((self.img_size[1] - 1) / 2)).astype('int64')
            maksed_depth = np.linalg.norm(masked_points[:, :3], 2, axis=1).reshape(-1, 1)

            masked_pcd_features = np.hstack((maksed_depth, masked_points[:,:4]))

            proj_points = np.zeros((self.img_size[0], self.img_size[1], 5)) # [depth,]得到points的feature映射在img的坐标上
            proj_points[masked_h, masked_w] = masked_pcd_features
            proj_points=proj_points.transpose((2, 0, 1))
            proj_labels = np.zeros((self.img_size[0], self.img_size[1], 1))  # [depth,]得到points的feature映射在img的坐标上
            proj_labels[masked_h, masked_w] = masked_labels.reshape(-1,1)
            proj_labels=proj_labels.transpose((2, 0, 1))

            image = cv.resize(image, (640, 512))
            image_input = image.transpose((2, 0, 1))

            # print(proj_points.shape, proj_labels.shape, image_input.shape)

            return {"pcd_feature":proj_points,
                    "pcd_labels":proj_labels,
                    "img_feature":image_input}

        elif self.split == 'val':
            with open(self.val_dataset_files[index], 'rb') as f:
                val_data = pickle.load(f)

            points= val_data['points'].numpy().squeeze()
            labels = val_data['labels'].numpy().squeeze()
            image = val_data['camera_img'].numpy().squeeze() # only front camera

            image = cv.resize(image, (640, 512))
            image_input = image.transpose((2, 0, 1))

            masked_points = points[points[:, -1] == -1.0]
            masked_labels = labels[points[:, -1] == -1.0]

            masked_h = np.floor((masked_points[:, 5] + 1) * ((self.img_size[0] - 1) / 2)).astype('int64')
            masked_w = np.floor((masked_points[:, 6] + 1) * ((self.img_size[1] - 1) / 2)).astype('int64')
            maksed_depth = np.linalg.norm(masked_points[:, :3], 2, axis=1).reshape(-1, 1)

            masked_pcd_features = np.hstack((maksed_depth, masked_points[:,:4]))

            proj_points = np.zeros((self.img_size[0], self.img_size[1], 5)) # [depth,]得到points的feature映射在img的坐标上
            proj_points[masked_h, masked_w] = masked_pcd_features
            proj_points=proj_points.transpose((2, 0, 1))
            proj_labels = np.zeros((self.img_size[0], self.img_size[1], 1))  # [depth,]得到points的feature映射在img的坐标上
            proj_labels[masked_h, masked_w] = masked_labels.reshape(-1,1)
            proj_labels=proj_labels.transpose((2, 0, 1))

            return {"pcd_feature":proj_points,
                    "pcd_labels":proj_labels,
                    "img_feature":image_input}

        # elif self.split == 'test':
        #     test_file = np.load(self.test_dataset_files[index])
        #     return test_file

    def __len__(self):
        if self.split == 'train':
            return len(self.train_dataset_files)
        elif self.split == 'val':
            return len(self.val_dataset_files)
        elif self.split == 'test':
            return len(self.val_dataset_files)

if __name__ == "__main__":
    dataset_root = '/usb/hd2t/new_nusc/'

    trainset = new_nuScenes(dataset_root, 'train')
    valset = new_nuScenes(dataset_root, 'val')
    testset = new_nuScenes(dataset_root, 'val')

    print(trainset.__len__())
    print(valset.__len__())
    # print(trainset[10]['pcd_labels'].count_nonzero())
    # print(valset[10]['pcd_labels'].count_nonzero())

    print(np.count_nonzero(trainset[10]['pcd_labels']))
    print(np.max(trainset[10]['pcd_labels']),np.min(trainset[10]['pcd_labels']))
    print(np.count_nonzero(trainset[10]['pcd_feature']))
    print(np.count_nonzero(trainset[10]['img_feature']))

    print(np.count_nonzero(valset[10]['pcd_labels']))
    print(np.max(valset[10]['pcd_labels']), np.min(valset[10]['pcd_labels']))
    print(np.count_nonzero(valset[10]['pcd_feature']))
    print(np.count_nonzero(valset[10]['img_feature']))
