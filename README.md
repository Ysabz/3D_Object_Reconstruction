# 3D Object Reconstruction with Multi-View RGB-D Images

With RGB-D cameras, we can get multiple RGB and Depth images and convert them to point clouds easily. Leveraging this, we can reconstruct single object with multi-view RGB and Depth images. To acheive this, point clouds from multi-view need to be registered. This task is also known as **point registration"**, whose goal is to find transfomration matrix between source and target point clouds. The alignment consists of two sub-alignment, Initial alignment and Alignment refinement. 

## RGB-D Camera Spec
- Model: Intel realsense D415
- Intrinsic parameters in 640x480 RGB, Depth image.<br> 
```
K = [[597.522, 0.0, 312.885],
     [0.0, 597.522, 239.870],
     [0.0, 0.0, 1.0]]
```
## Requirements
- Open3D
- Pyrealsense2
- OpenCV
- Numpy

## Align RGB and Depth Image & Depth Filtering
Due to the different position of RGB and Depth lens, aligning them should be done to get exact point clouds. This project used alignment function offered by pyrealsense2 package. Raw depth data are so noisy that depth filtering should be needed. Pyrealsense2 library, developed by Intel, offers filtering methods of depth images. In this project, spatial-filter was used that smoothes noise and preserves edge components in depth images. 

## Pre-Process Point Clouds
Single object might be a part of the scanned data. In order to get points of interested objects, pre-processing should be implemented. Plane-removal, outlier-removal, DBSCAN clustering were executed to extract object. Open3D offers useful functions to filter points. 

## Initial Alignment - Feature based 
Initial alignment can be acheived through finding transformation matrix between feature points, found by SIFT. The position of 3D points can be estimated with Back-Projection and Depth from depth images. Transformation matrix can be estimated with 3D corresponding feature points from souce and target point clouds, with RANSAC procedure. In order to find correspondeces in object area, extracted object 3D points were reprojected and the bounding box obtained to filter correspondeces out of object. 

## Alignment Refinement - ICP based
With ICP algorithm implemented in Open3D, refine initial transformation matrix. In this project, Point-to-Plane ICP method was used.

## Pose Graph Optimization
With optimzied pose graph, 

## Results <br>
The object was reconstructed with multiple different view of RGB-D Images. <br>

The reconstructed point clouds is below. <br>
<img src="https://user-images.githubusercontent.com/50229148/205788233-f9ee0b54-041c-4322-9d40-400a88220c9d.gif" width="500" height="300">
<img src="https://user-images.githubusercontent.com/50229148/205788281-585d8de4-1218-46ae-9d44-1afadef26e0a.gif" width="500" height="300">
