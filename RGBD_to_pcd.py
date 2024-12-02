import open3d as o3d
import numpy as np
import cv2
import os
import argparse
import glob
import numpy as np


# Depth scaling factor
depth_scaling_factor = 0.1  # Example scale (adjust as per your sensor)
fx, fy = 525, 525
cx, cy = 319.5, 239.5


# This function creates a point cloud for a given frame
def make_point_cloud(i, obj, depth_thresholds, width_range, height_range):

    # Read the images
    img = cv2.imread(f'./train/{obj}/rgb/COLOUR_STREAM{i}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    depth = o3d.io.read_image(f'./train/{obj}/depth/DEPTH_STREAM{i}.png')
    depth = np.asarray(depth, np.float32)

    # Filter image points based on depth
    idx = np.where((depth < depth_thresholds[0]) | (depth > depth_thresholds[1]))
    depth[idx] = 0
    
    # Create the normalized point cloud
    original_pcd = o3d.geometry.PointCloud()
    original_pcd_pos = []
    original_pcd_color = []

    for v in range(img.shape[0]): # height
        for u in range(img.shape[1]): # width

            
            # If the point is not in the cropped region, do not include it
            if (u < width_range[0]) or (u > width_range[1]):
                continue

            if (v < height_range[0]) or (v > height_range[1]):
                continue
            
            # Normalized image plane -> (u, v, 1) * z = zu, zv, z
            z = depth[v][u] * depth_scaling_factor # mm
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            original_pcd_pos.append([x, y, z])
            original_pcd_color.append(img[v][u] / 255)


    original_pcd_pos = np.array(original_pcd_pos, dtype=np.float32)
    original_pcd_color = np.array(original_pcd_color, dtype=np.float32)

    original_pcd.points = o3d.utility.Vector3dVector(original_pcd_pos)
    original_pcd.colors = o3d.utility.Vector3dVector(original_pcd_color)

    # Save point cloud
    o3d.io.write_point_cloud(f'./{obj}/{obj}{i}.pcd', original_pcd)
    #o3d.visualization.draw_geometries([original_pcd])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ob", help="Object Name (castard, box, spyderman, cap, kleenex)", required=True)

    # Used to filter the points based on depth
    parser.add_argument("-mid", help="Minimum Depth (Cap: 0, Kleenex: 0)", required=True)
    parser.add_argument("-mad", help="Maximum Depth (Cap: 750, Kleenex: 870)", required=True)

    # Used to crop the object within the image
    # Width: left to right
    # Height: bottom to top
    parser.add_argument("-miw", help="Minimum width (Cap: 215, Kleenex: 250)", required=True)
    parser.add_argument("-maw", help="Maximum width (Cap: 420, Kleenex: 380)", required=True)
    parser.add_argument("-mih", help="Minimum height (Cap: 150, Kleenex: 150)", required=True)
    parser.add_argument("-mah", help="Maximum height (Cap: 340, Kleenex: 310)", required=True)
    
    args = parser.parse_args()
    obj = args.ob
    min_depth = int(args.mid)
    max_depth = int(args.mad)
    min_width = int(args.miw)
    max_width = int(args.maw)
    min_height = int(args.mih)
    max_height = int(args.mah)

    os.makedirs(f'./{obj}', exist_ok=True)
    
    # Count the number of frames we have for this object if we haven't specified a last frame
    frames = glob.glob(f'./train/{obj}/rgb/*.png')
    nb_frames = len(frames)

    for i in range(1, nb_frames + 1):
        make_point_cloud(i, obj, [min_depth, max_depth], [min_width, max_width], [min_height, max_height])