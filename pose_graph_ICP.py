import numpy as np
import open3d as o3d
import copy


# Load point clouds
def load_point_clouds(voxel_size=0.0, pcds_paths=None):
    pcds = []
    print('len demo_icp_pcds_paths:', len(pcds_paths))
    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    for path in pcds_paths:
        pcd = o3d.io.read_point_cloud(path)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds


# Point to Plane ICP
# Can investigate Point to Point
def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine):

    init_trans=np.identity(4)

    # Required by TransformationEstimationPointToPlane
    source.estimate_normals()
    target.estimate_normals()

    # Get a rough estimate of the transformation
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # Refine the transformation
    # (from Open3D: requires an initial transfomration that roughly aligns the source and the target)
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    
    transformation_icp = icp_fine.transformation
    return transformation_icp



def sequential_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):

    # Initialize with first point cloud
    combined = copy.deepcopy(pcds[0])  # Start with first frame
    
    # For each new frame
    for i in range(1, len(pcds)):
        print(f'Aligning frame {i} to combined cloud')
        
        # Compute the transformation to go from the source (combined) to the target (new)
        transformation_icp= pairwise_registration(
            combined,
            pcds[i],  # Current frame
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine
        )
        
        # Transform the combined point clouds to stitch the target
        combined.transform(transformation_icp)
        combined += pcds[i]
        
        # Downsample the points
        combined = combined.voxel_down_sample(voxel_size=0.001)

        
        cl, _ = combined.remove_radius_outlier(
            nb_points=200,    
            radius=0.1       
        )
        
        
        cl, _ = combined.remove_statistical_outlier(
            nb_neighbors=200,    
            std_ratio=2.0       
        )
        
        combined = cl
        
        
    return combined



if __name__ == "__main__":

    #object = "new_box2"

    depth_path = ['./train/spyderman2/depth/align_test_depth%d.png' % i for i in range(1, 21)]
    rgb_path = ['./train/spyderman2/rgb/align_test%d.png' % i for i in range(1, 21)]
    pcds_paths = ['./spyderman/spyderman%d.pcd' % i for i in range(1, 21)]


    # Define voxel size to Downsample
    voxel_size = 0.001
    pcds_down = load_point_clouds(voxel_size, pcds_paths)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        combined_cloud = sequential_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)

    o3d.visualization.draw_geometries([combined_cloud])