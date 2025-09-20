# -*- coding: utf-8 -*-
"""
This script visualizes a point cloud stored in a .ply file using Open3D.

Usage:
python visualize_point_cloud.py <path_to_ply_file>

Example:
python visualize_point_cloud.py output/sparse_cloud.ply
"""
import open3d as o3d
import sys

def visualize_ply(file_path):
    """Loads and visualizes a .ply file."""
    print(f"Loading point cloud from {file_path}...")
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"Error: Could not read file {file_path}")
        print(e)
        return

    if not pcd.has_points():
        print("Error: The point cloud is empty or could not be loaded.")
        return

    print("Visualizing point cloud. Press 'Q' in the window to close.")
    
    # Create a visualizer window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud')
    
    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)
    
    # Improve rendering options
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]  # Dark background
    opt.point_size = 2.0
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_point_cloud.py <path_to_ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_ply(ply_file)
