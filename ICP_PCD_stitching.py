import numpy as np
import open3d as o3d
import copy
import argparse
import glob
import os
from pathlib import Path


# Load point clouds
def load_point_clouds(voxel_size, pcds_paths):
    pcds = []
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds


# Point to Plane ICP
# Can investigate Point to Point
def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine,
                          max_iteration):

    init_trans=np.identity(4)

    # Required by TransformationEstimationPointToPlane
    source.estimate_normals()
    target.estimate_normals()

    # Get a rough estimate of the transformation
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # Refine the transformation
    # (from Open3D: requires an initial transformation that roughly aligns the source and the target)
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    
    transformation_icp = icp_fine.transformation
    return transformation_icp


# Sequentially stitch the point clouds using the transformation computed by ICP
def sequential_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine, voxel_size_stitching,
                            max_iteration, nb_points, radius, nb_neighbors, std_ratio):

    # Initialize with first point cloud
    combined = copy.deepcopy(pcds[0])  # Start with first frame
    
    # For each new frame
    for i in range(1, len(pcds)):
        print(f'Aligning frame {i} to combined cloud')
        
        # Compute the transformation to go from the source (combined) to the target (new)
        transformation_icp= pairwise_registration(
            combined, # Source frame is the combined point clouds
            pcds[i],  # Target frame is the new frame
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine,
            max_iteration
        )
        
        # Transform the combined point clouds and stitch the target
        combined.transform(transformation_icp)
        combined += pcds[i]
        
        # Downsample the points using voxel
        combined = combined.voxel_down_sample(voxel_size=voxel_size_stitching)

        # Downsample the points using remove_radius_outlier
        cl, _ = combined.remove_radius_outlier(
            nb_points=nb_points,    
            radius=radius       
        )
        
        # Downsample the points using remove_statistical_outlier
        cl, _ = combined.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,    
            std_ratio=std_ratio       
        )
        
        combined = cl
        
    return combined



if __name__ == "__main__":

    # Script arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("-ob", help="Object Name (castard, box, spyderman)", required=True)
    parser.add_argument("-f", help="(Optional) First Frame", default=1)
    parser.add_argument("-l", help="(Optional) Last Frame", default=-1)
    parser.add_argument("-o", help="Output Directory Path", default="./results")

    # Algorithm hyperparameters
    parser.add_argument("-vl", help="Voxel size when loading the point clouds", default=0.001)
    parser.add_argument("-vs", help="Voxel size after stitching the point clouds", default=0.001)
    parser.add_argument("-cs", help="Maximum correspondence distance coarse scale factor", default=15)
    parser.add_argument("-fs", help="Maximum correspondence distance fine scale factor", default=1)
    parser.add_argument("-mi", help="Maximum number of iterations of ICP for the refinement", default=200)
    parser.add_argument("-np", help="Number of points in remove_radius_outlier", default=20)
    parser.add_argument("-r", help="Radius in remove_radius_outlier", default=0.1)
    parser.add_argument("-nn", help="Number of neighbors in remove_statistical_outlier", default=20)
    parser.add_argument("-sr", help="Std_ratio in remove_statistical_outlier", default=1.0)
    
    args = parser.parse_args()
    
    # Retrieve arguments
    obj = args.ob
    first_frame = int(args.f) # If none specified, start with first frame
    last_frame = int(args.l) # If none specified, end with last frame
    output_dir = args.o

    # Retrieve hyperparameter values
    voxel_size_loading = args.vl
    voxel_size_stitching = args.vs
    coarse_scale_factor = args.cs
    fine_scale_factor = args.fs
    max_iteration = args.mi
    nb_points = args.np
    radius = args.r
    nb_neighbors = args.nn
    std_ratio = args.sr

    # Count the number of frames we have for this object if we haven't specified a last frame
    if (last_frame == -1):
        frames = glob.glob(f'./train/{obj}/rgb/*.png')
        last_frame = len(frames)

    # Corresponding point clouds for the stitching we want to do
    pcds_paths = [f'./pcd_new/{obj}/{obj}' + '%d.pcd' % i for i in range(first_frame, last_frame + 1)]

    # Load the point clouds
    pcds_down = load_point_clouds(voxel_size_loading, pcds_paths)

    # ICP stitching
    max_correspondence_distance_coarse = voxel_size_stitching * coarse_scale_factor
    max_correspondence_distance_fine = voxel_size_stitching * fine_scale_factor
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        combined_cloud = sequential_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine, voxel_size_stitching,
                                                 max_iteration, nb_points, radius, nb_neighbors, std_ratio)

    # Visualize the stitched point clouds and store the output
    o3d.visualization.draw_geometries([combined_cloud])
    os.makedirs('./spyderman', exist_ok=True)
    o3d.io.write_point_cloud(output_dir + f"/{obj}_icp_stitching.pcd", combined_cloud) # (Filename must be a string, not a Path)