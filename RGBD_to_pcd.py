import open3d as o3d
import numpy as np
import cv2
import os


# Intel RealSense D415
depth_scaling_factor = 1000
focal_length = 597.522  ## mm
img_center_x = 312.885
img_center_y = 239.870

def make_point_cloud(i):
    # Read the images
    img = cv2.imread(f'./train/spyderman2/rgb/align_test{i}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = o3d.io.read_image(f'./train/spyderman2/depth/align_test_depth{i}.png')
    depth = np.asarray(depth, np.float32)

    # Remove the image points with a depth further than a certain value
    # (belongs to the background)
    threshold = 700 # 500
    idx = np.where(depth > threshold)
    depth[idx] = 0

    
    threshold2 = 500
    idx2 = np.where(depth < threshold2)
    depth[idx2] = 0
    

    # Create the normalized point cloud
    original_pcd = o3d.geometry.PointCloud()
    original_pcd_pos = []
    original_pcd_color = []

    for v in range(img.shape[0]): # height
        for u in range(img.shape[1]): # width
            
            
            if (u > 460) or (u < 200):
                continue
            
            
            '''
            if (v > 420): # 405
                continue
            '''
            

            # Normalized image plane -> (u, v, 1) * z = zu, zv, z
            z = depth[v][u] / depth_scaling_factor # mm
            x = (u - img_center_x) * z / focal_length
            y = (v - img_center_y) * z / focal_length

            original_pcd_pos.append([x, y, z])
            original_pcd_color.append(img[v][u] / 255)


    original_pcd_pos = np.array(original_pcd_pos, dtype=np.float32)
    original_pcd_color = np.array(original_pcd_color, dtype=np.float32)

    original_pcd.points = o3d.utility.Vector3dVector(original_pcd_pos)
    original_pcd.colors = o3d.utility.Vector3dVector(original_pcd_color)


    # Save point cloud
    o3d.io.write_point_cloud(f'./spyderman/spyderman{i}.pcd', original_pcd)
    #o3d.visualization.draw_geometries([original_pcd])



if __name__ == "__main__":
    os.makedirs('./spyderman', exist_ok=True)
    #make_point_cloud(1)
    
    
    for i in range(1, 23):
        make_point_cloud(i)
    
    
    