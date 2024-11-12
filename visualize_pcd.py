import open3d as o3d
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ob", help="Object to visualize", required=True)
    args = parser.parse_args()
    obj = args.ob

    pcd = o3d.io.read_point_cloud(f"./results/{obj}_icp_stitching.pcd")
    o3d.visualization.draw_geometries([pcd])