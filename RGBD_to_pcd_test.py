import open3d as o3d
import numpy as np
import cv2
import os
import glob

# Depth scaling factor
depth_scaling_factor = 0.1  # Example scale (adjust as per your sensor)

# Camera intrinsics (adjust these values based on your setup)
fx, fy = 1075.65091572, 1073.90347929
cx, cy = 641.068883438, 507.72159802

# Original and new resolutions
original_width, original_height = 1280, 1024
new_width, new_height = 320, 240

# Scaling factors
scale_x = new_width / original_width
scale_y = new_height / original_height

# Adjusted intrinsics
fx = fx * scale_x
fy = fy * scale_y
cx = cx * scale_x
cy = cy * scale_y

# Function to create a point cloud from RGB and Depth images
def make_point_cloud(rgb_path, depth_path, output_folder, frame_number):
    # Read RGB and Depth images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = o3d.io.read_image(depth_path)
    depth = np.asarray(depth, dtype=np.float32)   # Scale to mm

    # Mask invalid depths (e.g., 0 or very large values)
    valid_depth = depth[(depth > 0) & (depth < np.inf)]

    # Compute min and max depth
    # min_depth = valid_depth.min()
    # max_depth = valid_depth.max()
    min_depth = 11
    max_depth = 50
    idx = np.where((depth < min_depth) | (depth > max_depth))
    depth[idx] = 0
    print(f"Min Depth: {min_depth}, Max Depth: {max_depth}")

    # Define crop region
    min_width = 110
    max_width = 170
    min_height = 100
    max_height = 150

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    points = []
    colors = []

    for v in range(min_height, max_height):  # Cropped height range
        for u in range(min_width, max_width):  # Cropped width range
            # Normalized image plane -> (u, v, 1) * z = zu, zv, z
            z = depth[v][u] * depth_scaling_factor # mm
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])
            colors.append(rgb[v][u] / 255)

    # Assign points and colors to the point cloud
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Save the point cloud
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"frame_{frame_number:03d}.pcd")
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved: {output_path}")

# Main function
def process_frames(rgb_folder, depth_folder, output_folder, frame_interval=20):
    # List RGB and Depth images (assuming filenames match in order)
    rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.png")))

    # print(len(rgb_files))
    # print(len(depth_files))

    # # Ensure there are equal numbers of RGB and Depth images
    # assert len(rgb_files) == len(depth_files), "RGB and Depth image counts do not match!"

    # Process every nth frame
    for i in range(0, len(rgb_files), frame_interval):
        rgb_path = rgb_files[i]
        depth_path = depth_files[i]
        print(rgb_path)
        print(depth_path)
        make_point_cloud(rgb_path, depth_path, output_folder, frame_number=i // frame_interval)

# Define paths
rgb_folder = "./data/box/rgb/COLOUR_STREAM/"  # Update with your RGB folder path
depth_folder = "./data/box/depth/DEPTH_STREAM/"  # Update with your Depth folder path
output_folder = "./test/box2/"  # Folder to save point clouds

# Process and save point clouds
process_frames(rgb_folder, depth_folder, output_folder, frame_interval=20)
