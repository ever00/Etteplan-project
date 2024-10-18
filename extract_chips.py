import open3d as o3d
import numpy as np
import os
import sys

CLOUD_PATH = sys.argv[1]
OUTPUT_FOLDER = sys.argv[3]
base_filename = os.path.splitext(os.path.basename(CLOUD_PATH))[0]
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, base_filename + '_')

DOWNSAMPLE_FACTOR = int(sys.argv[2])
THRESHOLD_FILTER_LARGE_PARTS = 650000
TRIM_TOP = 0.14  
TRIM_BOTTOM = 0.02
SOR_NB_NEIGHBORS = 500
SOR_STD_RATIO = 5 
RANSAC_DISTANCE_THRESHOLD = 120
RANSAC_N = 2000
RANSAC_NUM_ITERATIONS = 1000
    

def split_chips(pcd):
    # Convert the point cloud to numpy arrays for easier manipulation
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Get the min and max bounds of the point cloud
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Calculate the range along each axis
    ranges = max_bound - min_bound
    longest_axis_index = np.argmax(ranges)  # Find the index of the longest axis

    # Define midpoints for partitioning along the longest axis
    midpoints = np.linspace(min_bound[longest_axis_index], max_bound[longest_axis_index], num=5)

    # Split the point cloud into 4 parts using the reverse order
    masks = [
        points[:, longest_axis_index] >= midpoints[3],  # Part 1
        (points[:, longest_axis_index] >= midpoints[2]) & (points[:, longest_axis_index] < midpoints[3]),  #Part 2
        (points[:, longest_axis_index] >= midpoints[1]) & (points[:, longest_axis_index] < midpoints[2]),  #Part 3
        points[:, longest_axis_index] < midpoints[1]  #Part 4
    ]

    parts = []
    for mask in masks:
        # Filter points and colors based on the mask
        part_points = points[mask]
        part_colors = colors[mask]

        # Create a new point cloud for each part
        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(part_points)
        part_pcd.colors = o3d.utility.Vector3dVector(part_colors)

        # Add the part to the list
        parts.append(part_pcd)

    return parts

def filter_large_parts(parts, min_points_threshold=THRESHOLD_FILTER_LARGE_PARTS):
    # Dictionary to store point clouds that meet the condition
    filtered_parts = {}

    for i, part in enumerate(parts):
        # Number of points in the current point cloud
        num_points = np.asarray(part.points).shape[0]
        
        # Check if the point cloud has more than the minimum number of points
        if num_points >= min_points_threshold:
            # Store the point cloud in the dictionary using its original index as the key
            filtered_parts[i] = part

    return filtered_parts


def cut_chip_edges(filtered_parts):
    # Iterate over the filtered parts (a dictionary with index and point clouds)
    trimmed_parts = {}
    
    for index, part_pcd in filtered_parts.items():
        # Convert the point cloud to numpy arrays for easier manipulation
        part_points = np.asarray(part_pcd.points)
        part_colors = np.asarray(part_pcd.colors)
        
        # Get the min and max bounds of the point cloud
        min_bound = part_points.min(axis=0)
        max_bound = part_points.max(axis=0)
        ranges = max_bound - min_bound
        longest_axis_index = np.argmax(ranges)  # Find the index of the longest axis

        # Define endpoints for trimming along the longest axis
        endpoints = np.linspace(min_bound[longest_axis_index], max_bound[longest_axis_index], num=2)
        
        # Set the upper and lower ratios for trimming the edges
        uppernumberratio = TRIM_TOP  # Percentage of the top part to trim
        lowernumberratio = TRIM_BOTTOM  # Percentage of the bottom part to trim

        # Create a mask to filter points and colors
        mask = (
            (endpoints[0] + (endpoints[1] - endpoints[0]) * lowernumberratio) < part_points[:, 1]
        ) & (part_points[:, 1] < endpoints[1] - (endpoints[1] - endpoints[0]) * uppernumberratio)

        # Apply the mask to filter both points and colors
        trimmed_points = part_points[mask]
        trimmed_colors = part_colors[mask]

        # Create a new trimmed point cloud
        trimmed_pcd = o3d.geometry.PointCloud()
        trimmed_pcd.points = o3d.utility.Vector3dVector(trimmed_points)
        trimmed_pcd.colors = o3d.utility.Vector3dVector(trimmed_colors)

        # Add the trimmed point cloud back into a dictionary with the original index
        trimmed_parts[index] = trimmed_pcd
       
    return trimmed_parts

def SOR(chip):
    # Remove noise, SOR removes point outside main structure
    _, ind = chip.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO) # jessica_500,5
    inlier_cloud = chip.select_by_index(ind)
    outlier_cloud = chip.select_by_index(ind, invert=True)
    return inlier_cloud, outlier_cloud

def RANSAC(chip, distance_threshold=RANSAC_DISTANCE_THRESHOLD, ransac_n=RANSAC_N, num_iterations=RANSAC_NUM_ITERATIONS):
    # Segment the largest plane using RANSAC
    _, inliers = chip.segment_plane(distance_threshold, ransac_n, num_iterations)
    inlier_cloud = chip.select_by_index(inliers)
    outlier_cloud = chip.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud

def center_and_normalize(part):
    # Convert point cloud to numpy array
    points = np.asarray(part.points)

    # Step 1: Center the point cloud by subtracting the mean of the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Step 2: Normalize the point cloud
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    normalized_points = centered_points / max_distance

    # Create a new point cloud with the normalized points
    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)
    normalized_pcd.colors = part.colors  # Retain the original colors if needed

    return normalized_pcd


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Application: python file.py <namepcdfile.pcd> <downsample_factor> <outputfolder>")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(CLOUD_PATH)
    if DOWNSAMPLE_FACTOR > 0:
        downsampled_pcd = pcd.uniform_down_sample(DOWNSAMPLE_FACTOR)
    
    #Step2: Split in 4 parts
    parts = split_chips(downsampled_pcd)
    #Step3: Filter out empty/manipulated  parts
    filtered_parts = filter_large_parts(parts)
    # Check if there are any parts left in the dictionary

    #Step4: Cut off labels
    if len(filtered_parts) == 0:
        sys.exit()

    trimmed_parts = cut_chip_edges(filtered_parts)

    #Step5: Noise Removal
    ransac_results = {}

    for index, part_pcd in trimmed_parts.items():
        inlier_cloud, _ = RANSAC(part_pcd)
            
        ransac_results[index] = {
            'inliers': inlier_cloud,
        }

    
    sor_results = {}

    # Iterate through RANSAC results
    for index, result in ransac_results.items():
        inliers = result['inliers']
        inlier_cloud_sor, _ = SOR(inliers)
        
        sor_results[index] = {
            'inliers_after_sor': inlier_cloud_sor,
        }

    #Step6: Normalize
    for index, result in sor_results.items():
        inliers = result['inliers_after_sor']

        # Center and normalize each inlier point cloud
        normalized_pcd = center_and_normalize(inliers)
        # Speichern der normalisierten Punktwolke
        output_filename = f"{OUTPUT_PATH}_part_{index+1}_downsample_{DOWNSAMPLE_FACTOR}.ply"
        o3d.io.write_point_cloud(output_filename, normalized_pcd)



        











