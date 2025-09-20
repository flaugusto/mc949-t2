# -*- coding: utf-8 -*-
"""
This script implements a simplified sequential Structure from Motion (SfM)
pipeline to generate a sparse point cloud from a folder of images.

It extends the two-view concept by incrementally adding new views:
1.  Initialize the 3D point cloud and camera poses with the first two images.
2.  For each subsequent image:
    a. Match features with the previous image.
    b. Estimate the relative pose (R, t) between the new and previous camera.
    c. Compose the relative pose with the absolute pose of the previous camera
       to get the new camera's absolute pose in the world frame.
    d. Triangulate new 3D points and add them to the global point cloud.

NOTE: This is a simplified approach and is prone to drift (accumulating errors)
because it lacks a global optimization step like Bundle Adjustment. For robust
results on larger datasets, using a tool like COLMAP is highly recommended.

Dependencies:
- opencv-contrib-python (for SIFT)
- numpy
"""
import cv2
import numpy as np
import os
import glob

# --- Configuration ---
# Path to the folder containing your images
IMAGE_FOLDER = 'images/'
OUTPUT_FOLDER = 'output'
OUTPUT_FILENAME = 'sequential_sparse_cloud.ply'

# Camera intrinsic parameters (initialized after loading the first image)
K = None

# SIFT and FLANN parameters
SIFT_FEATURES = 4000
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
LOWE_RATIO = 0.75

# --- Helper Functions from previous script (with minor changes) ---

def load_images_from_folder(folder_path):
    """Loads all images from a folder in sorted order."""
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')))
    images = [cv2.imread(path) for path in image_paths]
    images = [img for img in images if img is not None]
    if len(images) < 2:
        print(f"Error: Need at least 2 images to process, but found {len(images)} in {folder_path}")
        exit()
    print(f"Loaded {len(images)} images from {folder_path}")
    return images

def extract_features(image, n_features=SIFT_FEATURES):
    """Detects and describes features in an image using SIFT."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """Matches features between two sets of descriptors using FLANN."""
    if descriptors1 is None or descriptors2 is None or len(descriptors1) < 2 or len(descriptors2) < 2:
        return []
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES),
        dict(checks=FLANN_CHECKS)
    )
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    # Lowe's ratio test
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
    return good_matches

def estimate_pose(kp1, kp2, matches, K):
    """Estimates the camera pose (R, t) from matches."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    if len(pts1) < 5 or len(pts2) < 5:
        return None, None, None, None

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, pts1[mask_pose.ravel() == 1], pts2[mask_pose.ravel() == 1]

def triangulate_points(P1, P2, pts1, pts2):
    """Triangulates 3D points from 2D correspondences and projection matrices."""
    pts1_hom = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_hom = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]
    
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_hom.T, pts2_hom.T)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
    return points_3d.T

def save_ply(points_3d, filename):
    """Saves a 3D point cloud to a .ply file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    header = f"""ply
format ascii 1.0
element vertex {len(points_3d)}
property float x
property float y
property float z
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, points_3d, fmt='%f %f %f')
    print(f"Sparse point cloud saved to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting sequential sparse reconstruction...")

    # 1. Load all images
    images = load_images_from_folder(IMAGE_FOLDER)
    h, w, _ = images[0].shape

    # Define camera matrix K using the first image's dimensions
    if K is None:
        FOCAL_LENGTH = w * 1.2 
        K = np.array([
            [FOCAL_LENGTH, 0, w / 2],
            [0, FOCAL_LENGTH, h / 2],
            [0, 0, 1]
        ])
        print("Using an estimated camera intrinsic matrix K.")

    # 2. Process all images sequentially
    all_points_3d = []
    
    # --- Initialization with the first pair ---
    print("\n--- Initializing with the first two views ---")
    kp1, des1 = extract_features(images[0])
    kp2, des2 = extract_features(images[1])
    
    matches = match_features(des1, des2)
    print(f"Found {len(matches)} matches between image 0 and 1.")
    
    # Estimate pose for the first pair
    R_rel, t_rel, inlier_pts1, inlier_pts2 = estimate_pose(kp1, kp2, matches, K)
    
    if R_rel is None:
        print("Could not initialize with the first two images. Exiting.")
        exit()

    # The first camera is at the origin
    R_abs_prev = np.eye(3)
    t_abs_prev = np.zeros((3, 1))
    
    # The second camera's pose is the relative pose from the first
    R_abs_curr = R_rel
    t_abs_curr = t_rel

    # Triangulate initial points
    P1 = K @ np.hstack((R_abs_prev, t_abs_prev))
    P2 = K @ np.hstack((R_abs_curr, t_abs_curr))
    initial_points_3d = triangulate_points(P1, P2, inlier_pts1, inlier_pts2)
    all_points_3d.append(initial_points_3d)

    print(f"Initialized with {len(initial_points_3d)} points.")

    # --- Incremental reconstruction for the rest of the images ---
    kp_prev, des_prev = kp2, des2
    
    for i in range(2, len(images)):
        print(f"\n--- Processing image {i} ---")
        
        # Extract features from the current image
        kp_curr, des_curr = extract_features(images[i])
        
        # Match features with the previous image
        matches = match_features(des_prev, des_curr)
        print(f"Found {len(matches)} matches between image {i-1} and {i}.")
        
        if len(matches) < 20:
            print("Not enough matches to continue. Stopping.")
            break
            
        # Estimate relative pose between previous and current camera
        R_rel, t_rel, inlier_pts_prev, inlier_pts_curr = estimate_pose(kp_prev, kp_curr, matches, K)
        
        if R_rel is None:
            print(f"Could not estimate pose for image {i}. Skipping.")
            continue

        # Update absolute pose of the current camera by composing transformations
        # P_world = P_prev * P_rel => T_world = T_prev * T_rel
        # R_abs_curr = R_rel @ R_abs_prev (Careful with order, depends on frame def)
        # For camera poses: T_curr_world = T_prev_world @ T_curr_prev
        t_abs_curr = t_abs_prev + R_abs_prev @ t_rel
        R_abs_curr = R_rel @ R_abs_prev
        
        # Triangulate new points
        P_prev = K @ np.hstack((R_abs_prev, t_abs_prev))
        P_curr = K @ np.hstack((R_abs_curr, t_abs_curr))
        new_points_3d = triangulate_points(P_prev, P_curr, inlier_pts_prev, inlier_pts_curr)
        all_points_3d.append(new_points_3d)
        
        print(f"Added {len(new_points_3d)} new points.")

        # Prepare for the next iteration
        R_abs_prev, t_abs_prev = R_abs_curr, t_abs_curr
        kp_prev, des_prev = kp_curr, des_curr
        
    # 3. Combine and save the final point cloud
    final_points = np.vstack(all_points_3d)
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    save_ply(final_points, output_path)
    
    print("\nPipeline finished.")
    print(f"Total points in cloud: {len(final_points)}")
    print(f"Run 'python visualize_point_cloud.py {output_path}' to see the result.")
