% Main script to import point clouds and run the ICP algorithm

clc;clear;close all;

addpath("./icp_matlab_exchange");

teapot = pcread("teapot.ply");
rotationAngles = [15,0,30];
translation = [-3 0.5 5];
transform1 = rigidtform3d(rotationAngles,translation);
trasformed_teapot = pctransform(teapot,transform1);


figure;
pcshow(teapot);
title("Teapot");
figure;
pcshow(trasformed_teapot);
title("Transfromed Teapot");

figure;
pcshow(teapot);
hold on;
pcshow(trasformed_teapot);
title("Teapot and Transfromed Teapot");

figure;
pcshowpair(teapot,trasformed_teapot,VerticalAxis="Y",VerticalAxisDir="down");

% % Load or import the point clouds (replace with actual data or file paths)
% P = load('source_point_cloud.mat');  % Replace with actual data loading method
% Q = load('target_point_cloud.mat');  % Replace with actual data loading method

% Ensure P and Q are in the correct format as Nx3 matrices
P = double(trasformed_teapot.Location);
Q = double(teapot.Location);

% Set parameters for the ICP algorithm
max_iterations = 100;
tolerance = 1e-3;

% Run the ICP algorithm
[R, t, P_aligned] = icp_algorithm(P, Q, max_iterations, tolerance);
aligned_teapot = pointCloud(P_aligned);

% Visualize the results
figure;
pcshow(teapot);
hold on;
pcshow(trasformed_teapot)
pcshow(aligned_teapot);
title('ICP Alignment Results');
grid on;
hold off; 


% Run the ICP algorithm from DTU
[Ricp,Ticp,ER,t] = icp(Q.', P.', 15);
Qicp = Ricp * P.' + repmat(Ticp, 1, length(Q));

% Visualize the results
figure;
pcshow(teapot);
hold on;
pcshow(trasformed_teapot)
pcshow(pointCloud(Qicp.'));
title('ICP Alignment Results DTU Method');
grid on;
hold off; 

function [R, t, P_aligned] = icp_algorithm(P, Q, max_iterations, tolerance)
% ICP Algorithm to align two 3D point clouds P and Q
% Inputs:
% P - Source point cloud (Nx3 matrix)
% Q - Target point cloud (Mx3 matrix)
% max_iterations - Maximum number of iterations for the algorithm
% tolerance - Convergence criterion for stopping (e.g., change in error)
% Outputs:
% R - Optimal rotation matrix (3x3)
% t - Optimal translation vector (3x1)
% P_aligned - Aligned source point cloud

% Initialize variables
N = size(P, 1);
prev_error = Inf;
P_aligned = P;  % Initially, the aligned source is the original source

for iter = 1:max_iterations
    % Step 1: Find the closest points in Q for each point in P_aligned
    closest_indices = zeros(N, 1);
    for i = 1:N
        distances = sqrt(sum((Q - P_aligned(i, :)).^2, 2));  % Compute Euclidean distance from P_aligned(i, :) to all points in Q
        [~, closest_indices(i)] = min(distances);  % Find the index of the closest point in Q
    end
    Q_matched = Q(closest_indices, :);  % Q points corresponding to the closest points to P_aligned

    % Step 2: Compute the centroids of P and Q (matched points)
    P_centroid = mean(P_aligned);
    Q_centroid = mean(Q_matched);

    % Step 3: Center the point sets
    P_centered = P_aligned - P_centroid;
    Q_centered = Q_matched - Q_centroid;

    % Step 4: Compute the cross-covariance matrix H
    H = P_centered' * Q_centered;

    % Step 5: Perform Singular Value Decomposition (SVD) on H
    [U, S, V] = svd(H);

    % Step 6: Compute the rotation matrix R
    R = V * U';

    % Ensure R is a valid rotation matrix (det(R) = 1)
    if det(R) < 0
        V(:, end) = -V(:, end);
        R = V * U';
    end
    % Step 7: Compute the translation vector t
    t = Q_centroid' - R * P_centroid';

    % Step 8: Apply the transformation to P
    P_aligned = (R * P')' + t';

    % Step 9: Check for convergence (change in mean squared error)
    mean_error = mean(sqrt(sum((P_aligned - Q_matched).^2, 2)));
    if abs(prev_error - mean_error) < tolerance
        disp(['Convergence reached at iteration ', num2str(iter)]);
        break;
    end
    prev_error = mean_error;
end

% Display final results
disp('Optimal rotation matrix R:');
disp(R);

disp('Optimal translation vector t:');
disp(t);

disp('Final mean squared error:');
disp(mean_error);

end