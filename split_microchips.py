import open3d as o3d
import numpy as np

# Visulize with open3d
pcd = o3d.io.read_point_cloud('C:/Users/jessi/OneDrive/Dokument/ProjectCourse/data/2024-09-02_130544_243301-11/raw/tray-b-4-a_L2.pcd')

# Convert to numpy array for easier manipulation
points = np.asarray(pcd.points)

min_bound = points.min(axis=0)
max_bound = points.max(axis=0)

ranges = max_bound - min_bound
longest_axis_index = np.argmax(ranges)  # Index of the longest axis

# Define midpoints for partitioning along the longest axis
midpoints = np.linspace(min_bound[longest_axis_index], max_bound[longest_axis_index], num=5)

# Define the four parts along the longest axis
if longest_axis_index == 0:  # Longest along x-axis
    parts = [
        points[(points[:, 0] < midpoints[1])],  # Part 1
        points[(points[:, 0] >= midpoints[1]) & (points[:, 0] < midpoints[2])],  # Part 2
        points[(points[:, 0] >= midpoints[2]) & (points[:, 0] < midpoints[3])],  # Part 3
        points[(points[:, 0] >= midpoints[3])],  # Part 4
    ]
elif longest_axis_index == 1:  # Longest along y-axis
    parts = [
        points[(points[:, 1] < midpoints[1])],  # Part 1
        points[(points[:, 1] >= midpoints[1]) & (points[:, 1] < midpoints[2])],  # Part 2
        points[(points[:, 1] >= midpoints[2]) & (points[:, 1] < midpoints[3])],  # Part 3
        points[(points[:, 1] >= midpoints[3])],  # Part 4
    ]
else:  # Longest along z-axis
    parts = [
        points[(points[:, 2] < midpoints[1])],  # Part 1
        points[(points[:, 2] >= midpoints[1]) & (points[:, 2] < midpoints[2])],  # Part 2
        points[(points[:, 2] >= midpoints[2]) & (points[:, 2] < midpoints[3])],  # Part 3
        points[(points[:, 2] >= midpoints[3])],  # Part 4
    ]

# Visualize each part
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # Colors for each part: red, green, blue, yellow
part_clouds = []

for i, part in enumerate(parts):
    if part.size > 0:  # Only create a PointCloud if there are points in the part
        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(part)
        part_pcd.paint_uniform_color(colors[i])  # Set the color for the part
        part_clouds.append(part_pcd)

# Draw the parts
o3d.visualization.draw_geometries(part_clouds, window_name="Partitioned Point Cloud Parts", width=800, height=600)