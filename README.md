# 3D Object Reconstruction with Multi-View RGB-D Images

In this project, we used RGBD data and the ICP algorithm to do 3D object reconstruction. Our setup was a fixed RGBD camera recording both color and depth images of a rotating object on a turntable. We used the camera's intrinsic parameters along with the depth information to derive the 3D point clouds. Then we sequentially stitched those point clouds using the ICP algorithm. The output is a colored 3D reconstruction of the object.

## RGB-D Camera Spec
- Model: Kinect-like (Washington RGB-D dataset)
- Intrinsic parameters in 640x480 RGB, Depth image.<br> 
```
K = [[525, 0.0, 319.5],
     [0.0, 525, 239.5],
     [0.0, 0.0, 1.0]]
```


