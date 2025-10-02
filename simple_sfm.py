# -*- coding: utf-8 -*-
"""
This script implements a basic two-view Structure from Motion (SfM) pipeline
to generate a sparse point cloud from two images.

It follows the main steps described in the academic assignment:
1.  Feature Detection and Description (using SIFT).
2.  Feature Matching (using FLANN matcher with Lowe's ratio test).
3.  Epipolar Geometry and Pose Estimation (recovering R and t).
4.  Triangulation to get 3D points.

The output is a .ply file that can be visualized with tools like MeshLab or Open3D.

Dependencies:
- opencv-contrib-python (for SIFT)
- numpy
"""
import cv2
import numpy as np
import os

# --- Configuration ---
# Paths to your images
IMAGE_PATH_1 = 'images/image1.jpg'
IMAGE_PATH_2 = 'images/image2.jpg'
OUTPUT_FOLDER = 'output'
OUTPUT_FILENAME = 'sparse_cloud.ply'

# Camera intrinsic parameters (you should use your camera's actual values)
# If you don't know them, you can estimate them or use reasonable defaults.
# For this example, we assume the principal point is at the center
# and we guess a focal length.
# FOCAL_LENGTH = 1200
# PRINCIPAL_POINT = (image_width / 2, image_height / 2)
# K = np.array([[FOCAL_LENGTH, 0, PRINCIPAL_POINT[0]],
#               [0, FOCAL_LENGTH, PRINCIPAL_POINT[1]],
#               [0, 0, 1]])

# If K is not known, we can try to estimate it. For this script,
# we will define it after loading the images.
K = None

# SIFT and FLANN parameters
SIFT_FEATURES = 4000
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
LOWE_RATIO = 0.75

# --- Helper Functions ---

def load_image(image_path):
    """Loads an image and checks if it exists."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        exit()
    return img

def extract_features(image, n_features=SIFT_FEATURES):
    """Detects and describes features in an image using SIFT."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """Matches features between two sets of descriptors using FLANN."""
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES),
        dict(checks=FLANN_CHECKS)
    )
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance:
            good_matches.append(m)
    return good_matches

def estimate_pose(kp1, kp2, matches, K):
    """
    Estimates the camera pose (Rotation and Translation) from matches.
    It computes the Essential matrix and recovers pose from it.
    """
    # Get the coordinates of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute the Essential Matrix
    # The RANSAC threshold can be adjusted
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None:
        print("Error: Could not compute the Essential Matrix. Check your images and matches.")
        return None, None, None, None

    # Recover the Pose (R, t) from the Essential Matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    # Filter matches using the pose mask
    inlier_matches = [m for i, m in enumerate(matches) if mask_pose[i]]

    return R, t, pts1[mask_pose.ravel() == 1], pts2[mask_pose.ravel() == 1]


def triangulate_points(R, t, pts1, pts2, K):
    """
    Triangulates 3D points from 2D correspondences and camera poses.
    """
    # Create projection matrices for both cameras
    # The first camera is at the origin
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # The second camera's pose is defined by R and t
    P2 = K @ np.hstack((R, t))

    # Convert points to the correct format for triangulatePoints
    # The function expects 2xN arrays of 2D points
    pts1_norm = cv2.undistortPoints(pts1.astype(np.float32), K, None)
    pts2_norm = cv2.undistortPoints(pts2.astype(np.float32), K, None)
    
    # Reshape to 2xN arrays
    pts1_reshaped = pts1_norm.reshape(-1, 2).T
    pts2_reshaped = pts2_norm.reshape(-1, 2).T
    
    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_reshaped, pts2_reshaped)

    # Convert from homogeneous to 3D Cartesian coordinates
    points_3d = points_4d_hom / points_4d_hom[3]
    return points_3d[:3].T

def save_ply(points_3d, filename):
    """Saves a 3D point cloud to a .ply file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create the PLY file header
    header = f"""ply
format ascii 1.0
element vertex {len(points_3d)}
property float x
property float y
property float z
end_header
"""
    # Write the header and points to the file
    with open(filename, 'w') as f:
        f.write(header)
        for point in points_3d:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Sparse point cloud saved to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting two-view sparse reconstruction...")

    # 1. Load Images
    img1 = load_image(IMAGE_PATH_1)
    img2 = load_image(IMAGE_PATH_2)
    h, w, _ = img1.shape
    
    # Define camera matrix K here if not known
    if K is None:
        # A reasonable guess for focal length
        FOCAL_LENGTH = w * 1.2 
        K = np.array([
            [FOCAL_LENGTH, 0, w / 2],
            [0, FOCAL_LENGTH, h / 2],
            [0, 0, 1]
        ])
        print("Using an estimated camera intrinsic matrix K.")

    # 2. Extract Features
    print("Step 1: Extracting SIFT features...")
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)
    print(f"Found {len(kp1)} features in image 1 and {len(kp2)} features in image 2.")

    # 3. Match Features
    print("Step 2: Matching features...")
    matches = match_features(des1, des2)
    print(f"Found {len(matches)} good matches after Lowe's ratio test.")
    
    # Optional: Visualize matches
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Matches", cv2.resize(img_matches, (w // 2, h // 2)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # 4. Estimate Pose
    print("Step 3: Estimating camera pose...")
    R, t, inlier_pts1, inlier_pts2 = estimate_pose(kp1, kp2, matches, K)
    
    if R is None:
        exit()
        
    print(f"Pose estimated successfully. Found {len(inlier_pts1)} inlier points.")

    # 5. Triangulate 3D Points
    print("Step 4: Triangulating 3D points...")
    points_3d = triangulate_points(R, t, inlier_pts1, inlier_pts2, K)
    print(f"Triangulated {len(points_3d)} points.")

    # 6. Save Point Cloud
    print("Step 5: Saving point cloud...")
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    save_ply(points_3d, output_path)

    print("\nPipeline finished.")
    print(f"Run 'python point_visualizer.py {output_path}' to see the result.")
