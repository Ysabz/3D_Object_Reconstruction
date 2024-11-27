import open3d as o3d
import glob

# Define the path to the folder containing your point cloud files
folder_path = "./test/box2/"
file_pattern = folder_path + "frame_*.pcd"  # Adjust if using a different extension like .ply
# folder_path = "./box/"
# file_pattern = folder_path + "box*.pcd"  # Adjust if using a different extension like .ply
# Get a list of all point cloud files
pcd_files = sorted(glob.glob(file_pattern))

# Load and visualize the point clouds
point_clouds = []

for file in pcd_files:
    print(f"Visualizing point cloud: {file}")
    pcd = o3d.io.read_point_cloud(file)
    o3d.visualization.draw_geometries([pcd])
