import os
import sys
import glob
import numpy as np
from plyflatten import plyflatten_from_plyfiles_list

import s2p
from s2p import common
# from tests_utils import data_path
from s2p import block_matching
# here = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.join(os.path.dirname(here)))
# from utils import s2p_mosaic

from s2p import ply
import pcl

def filter_ply_file(input_file, output_file):
   
    # Compute the point cloud x, y bounds
    points, comments  = ply.read_3d_point_cloud_from_ply(input_file)
    
    # Subtract the center value
    center = np.mean(points, axis=0)
    points -= center 

    # Select the first three columns corresponding to x, y, z
    points = points[:,:3]

    # Convert to pcl type
    cloud = pcl.PointCloud()
    cloud.from_array(points.astype(np.float32))
    
    # Create a statistical filter
    s_filter = cloud.make_statistical_outlier_filter()
    s_filter.set_mean_k(50)          # Set the number of points used to calculate the average
    s_filter.set_std_dev_mul_thresh(1.0)  # Set the threshold multiples of standard deviation

    # Perform filtering
    cloud_filtered = s_filter.filter()
    
        # Get the numpy array of points from the filtered cloud
    points_filtered = cloud_filtered.to_array()
    print("Number of points rows, columns:", points_filtered.shape)
    # Add the center value back to the points
   
    points_filtered += center[:3]
    print("Number of points rows, columns:", points_filtered.shape)
    # Save filtered point cloud
    pcl.save(cloud_filtered, output_file)
    
def filter_ply_file1(input_file, output_file):
    # Compute the point cloud x, y bounds
    points, comments  = ply.read_3d_point_cloud_from_ply(input_file)
    
    # Subtract the center value for the x, y, z coordinates, not for the color
    center = np.mean(points[:, :3], axis=0)
    points[:, :3] -= center 
    print("Number of points rows, columns:", points.shape)
  
    cloud = pcl.PointCloud_PointXYZRGBA([[point[0],
                                      point[1],
                                      point[2],
                                      (int(point[3]) << 16) | (int(point[4]) << 8) | int(point[5])] for point in points])
   
    # pcl.save(cloud, output_file, format='ply')
    # Load Point Cloud file
    # cloud = pcl.load_XYZRGB(output_file)

    # Create a statistical filter
    s_filter = cloud.make_statistical_outlier_filter()
    s_filter.set_mean_k(50)          # Set the number of points used to calculate the average
    s_filter.set_std_dev_mul_thresh(1.0)  # Set the threshold multiples of standard deviation

    # Perform filtering
    cloud_filtered = s_filter.filter()

    # Save filtered point cloud with color
    pcl.save(cloud_filtered, output_file, format='ply')
  

def convert64plyto32ply(input_file, output_file):
   
    # Compute the point cloud x, y bounds
    points, comments  = ply.read_3d_point_cloud_from_ply(input_file)
    
    # Subtract the center value
    center = np.mean(points, axis=0)
    
    points -= center 

    # Select the first three columns corresponding to x, y, z
    points = points[:,:3]

    # Convert to pcl type
    cloud = pcl.PointCloud()
    cloud.from_array(points.astype(np.float32))
    
    # Save filtered point cloud
    pcl.save(cloud, output_file)
     
# import open3d as o3d

# def statistical_outlier_filter(pcd_path, nb_neighbors=20, std_ratio=2.0):
#     """
#     对点云数据进行统计滤波处理。

#     参数:
#     - pcd_path: 点云数据文件的路径。
#     - nb_neighbors: 用于估计点云平均密度的临近点数量。
#     - std_ratio: 指定去除点云的比例，是标准差的倍数。

#     返回:
#     - 经过滤波处理的点云数据。
#     """
#     pcd = o3d.io.read_point_cloud(pcd_path)

#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
#                                              std_ratio=std_ratio)
#     return cl
   
def generate_dsm(config_file, absmean_tol=0.025, percentile_tol=1.):

    # TODO: this is ugly, and will be fixed once we'll have implemented a better
    # way to control the config parameters
    if 'out_crs' in s2p.cfg: del s2p.cfg['out_crs']

    test_cfg = s2p.read_config_file(config_file)
    s2p.main(test_cfg)

    outdir = test_cfg['out_dir']

#generate_dsm('/root/s2p/Data/ZY3_DLC/config.json', 0.025, 1)
# generate_dsm('/root/s2p/Data/config/gaofen_config.json', 0.025, 1)
# generate_dsm('/root/s2p/Data/config/gaofen_config_rect.json', 0.025, 1)

generate_dsm('/root/s2p/Data/config/zy3dsm_config.json', 0.025, 1)

# disp_map = '/root/s2p/Data/Output/zy3_rect/tiles/row_0002376_height_594/col_0009660_width_604/pair_1/rectified_disp.tif'
# disp_map_fill = '/root/s2p/Data/Output/zy3_rect/tiles/row_0002376_height_594/col_0009660_width_604/pair_1/rectified_disp_fill.tif'

# block_matching.insertDepth32f(disp_map, disp_map_fill)

# 对输入的点云数据进行滤波并保存结果
# filter_ply_file('/root/s2p/Data/Output/zy3_rect/tiles/row_0002376_height_594/col_0009660_width_604/cloud.ply', '/root/s2p/Data/Output/zy3_rect/tiles/row_0002376_height_594/col_0009660_width_604/cloud_filter.ply')

# convert64plyto32ply('/root/s2p/Data/Output/zy3_rect/tiles/row_0002376_height_594/col_0010868_width_604/cloud.ply',
#                     '/root/s2p/Data/Output/zy3_rect/tiles/row_0002376_height_594/col_0010868_width_604/pcloud.ply')

# 创建 Point Cloud 对象
# p = pcl.load("/root/s2p/Data/Output/zy3_rect/tiles/row_0002376_height_594/col_0010868_width_604/pcloud.ply")

# fil = p.make_statistical_outlier_filter()

