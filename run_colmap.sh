#!/bin/bash
#
# This script automates the full Structure from Motion (SfM) and
# Multi-View Stereo (MVS) pipeline using COLMAP's command-line interface.
#
# Usage:
# 1. Install COLMAP: https://colmap.github.io/install.html
# 2. Create a project folder (e.g., 'my_project').
# 3. Inside, create an 'images' folder with all your pictures.
# 4. Place this script inside 'my_project' and run it: ./run_colmap.sh

# --- Configuration ---
# Set the main project directory
PROJECT_PATH="."

# Set the path to the folder containing your images
IMAGE_PATH="$PROJECT_PATH/final_images"

# Set the path for the database
DATABASE_PATH="$PROJECT_PATH/database.db"

# --- Pipeline ---

# Exit on error
set -e

echo "------------------------------------"
echo "Starting COLMAP pipeline..."
echo "Project Path: $PROJECT_PATH"
echo "Image Path:   $IMAGE_PATH"
echo "------------------------------------"

# Step 1: Feature Extraction
# Extracts features from all images and stores them in a database.
echo "\n[COLMAP] Step 1: Feature Extraction..."
colmap feature_extractor \
   --database_path "$DATABASE_PATH" \
   --image_path "$IMAGE_PATH" \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model OPENCV \
   --SiftExtraction.use_gpu 1

# Step 2: Feature Matching
# Matches features between image pairs. Exhaustive matching is robust for smaller datasets.
echo "\n[COLMAP] Step 2: Feature Matching..."
colmap exhaustive_matcher \
   --database_path "$DATABASE_PATH" \
   --SiftMatching.use_gpu 1

# Step 3: Sparse Reconstruction (Structure from Motion)
# This is the core step that creates the sparse point cloud and estimates camera poses.
echo "\n[COLMAP] Step 3: Mapping / Sparse Reconstruction..."
# Create a directory for the sparse model
mkdir -p "$PROJECT_PATH/sparse"

colmap mapper \
   --database_path "$DATABASE_PATH" \
   --image_path "$IMAGE_PATH" \
   --output_path "$PROJECT_PATH/sparse"

echo "\n------------------------------------"
echo "Sparse reconstruction complete!"
echo "You can now view the sparse model by running:"
echo "colmap gui --database_path $DATABASE_PATH --import_path $PROJECT_PATH/sparse/0"
echo "------------------------------------"

# Optional Steps for Dense Reconstruction (uncomment to run)

# echo "\n[COLMAP] Step 4: Image Undistortion..."
# # Creates undistorted images and camera parameters needed for dense reconstruction
# mkdir -p "$PROJECT_PATH/dense"
#
# colmap image_undistorter \
#    --image_path "$IMAGE_PATH" \
#    --input_path "$PROJECT_PATH/sparse/0" \
#    --output_path "$PROJECT_PATH/dense" \
#    --output_type COLMAP
#
# echo "\n[COLMAP] Step 5: Dense Reconstruction (MVS)..."
# # Generates a dense point cloud from the sparse model and undistorted images
# colmap patch_match_stereo \
#    --workspace_path "$PROJECT_PATH/dense" \
#    --PatchMatchStereo.geom_consistency true
#
# echo "\n[COLMAP] Step 6: Dense Fusion..."
# # Fuses the depth maps into a final dense point cloud
# colmap stereo_fusion \
#    --workspace_path "$PROJECT_PATH/dense" \
#    --output_path "$PROJECT_PATH/dense/fused.ply"
#
# echo "\n------------------------------------"
# echo "Dense reconstruction complete!"
# echo "Dense point cloud saved to: $PROJECT_PATH/dense/fused.ply"
# echo "------------------------------------"
