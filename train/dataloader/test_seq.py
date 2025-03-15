import os
import sys
sys.path.append("unigarment/train")
sys.path.append("/home/user/DexGarmentLab-master/DexGarmentLab-master/unigarment")
sys.path.append("unigarment/train/dataloader")

import numpy as np
from utils import visualize_point_cloud
from base.config import Config

path = '/home/user/DexGarmentLab-master/DexGarmentLab-master/data/ls_tops/cd_processed/mesh_pcd/66_TCLC_Jacket162_obj/p_0.npz'
data = np.load(path)

kp_id = data['pcd_keypoints_id']
pcd_points = data['pcd_points']

visualize_point_cloud(pcd_points, kp_id)
# visualize_point_cloud(pcd_points, None)
for i in range(kp_id.shape[0]):
        visualize_point_cloud(pcd_points, kp_id)