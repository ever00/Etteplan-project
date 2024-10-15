import open3d as o3d
import numpy as np
import os

    
def split_chips(pcd):
    # Devide the pointcloud into 4 parts, one for each chip
    # Convert to numpy arrays for easier manipulation
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # Get the colors of the point cloud

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    ranges = max_bound - min_bound
    longest_axis_index = np.argmax(ranges)  # Index of the longest axis

    # Define midpoints for partitioning along the longest axis (5 evenly spaced points creating 4)
    midpoints = np.linspace(min_bound[longest_axis_index], max_bound[longest_axis_index], num=5)

    parts = [
        (points[(points[:, 1] < midpoints[1])], colors[(points[:, 1] < midpoints[1])]),  # Part 1
        (points[(points[:, 1] >= midpoints[1]) & (points[:, 1] < midpoints[2])], colors[(points[:, 1] >= midpoints[1]) & (points[:, 1] < midpoints[2])]),  # Part 2
        (points[(points[:, 1] >= midpoints[2]) & (points[:, 1] < midpoints[3])], colors[(points[:, 1] >= midpoints[2]) & (points[:, 1] < midpoints[3])]),  # Part 3
        (points[(points[:, 1] >= midpoints[3])], colors[(points[:, 1] >= midpoints[3])])]  # Part 4

    return parts


def remove_empty_chips(part_colors):
    # Detect empty chips
    chip = o3d.geometry.PointCloud()
    chip.colors = o3d.utility.Vector3dVector(part_colors)
    
    all_points_num = np.count_nonzero(part_colors)/3

    if all_points_num/8000000 > 0.9:
        return True
    else:
        return False
    

def cut_chip_edges(part_points, part_colors):
        # Cut the noise around the chip
        min_bound = part_points.min(axis=0)
        max_bound = part_points.max(axis=0)
        ranges = max_bound - min_bound
        longest_axis_index = np.argmax(ranges)  # Index of the longest axis
        
        min_bound[longest_axis_index], max_bound[longest_axis_index]
        
        endpoints = np.linspace(min_bound[longest_axis_index], max_bound[longest_axis_index], num=2)
        
        uppernumberratio=0.14 #approximate ratio of how much room numbers take up
        lowernumberratio=0.02
        
        part_points2=part_points
        
        part_points = part_points[((endpoints[0]+(endpoints[1]-endpoints[0])*lowernumberratio)<part_points[:, 1])&(part_points[:, 1] < endpoints[1]-(endpoints[1]-endpoints[0])*uppernumberratio)]
        part_colors = part_colors[((endpoints[0]+(endpoints[1]-endpoints[0])*lowernumberratio)<part_points2[:, 1])&(part_points2[:, 1] < endpoints[1]-(endpoints[1]-endpoints[0])*uppernumberratio)]
        
        return part_points, part_colors


def pc_from_numpy(part_points, part_colors):
    # Convert a NumPy array of shape (N, 3) to an Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(part_points)
    pcd.colors = o3d.utility.Vector3dVector(part_colors)  # Preserve the original colors
    return pcd


def downsample(pcd):
    # Downsample pointcloud using uniform_down_sample
    downsampled_pcd = pcd.uniform_down_sample(10)

    # Use voxel downsampling instead when computepower is available 
    #downsampled_pcd = pcd.voxel_down_sample(voxel_size=2*avg_dist)
    return downsampled_pcd


def SOR(chip):
    # Remove noise, SOR removes point outside main structure
    _, ind = chip.remove_statistical_outlier(nb_neighbors=500, std_ratio=15)
    inlier_cloud = chip.select_by_index(ind)
    outlier_cloud = chip.select_by_index(ind, invert=True)
    return inlier_cloud, outlier_cloud


def RANSAC(chip, distance_threshold=120, ransac_n=2000, num_iterations=1000):
    # Segment the largest plane using RANSAC
    _, inliers = chip.segment_plane(distance_threshold, ransac_n, num_iterations)
    inlier_cloud = chip.select_by_index(inliers)
    outlier_cloud = chip.select_by_index(inliers, invert=True)
    return inlier_cloud, outlier_cloud


def write_to_file(chip, filename, i):
    output_dir = 'C:/Users/jessi/OneDrive/Dokument/ProjectCourse/output'

    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{output_dir}/{filename}_part_{i + 1}.pcd"
    # Save the part to a PCD file
    o3d.io.write_point_cloud(output_filename, chip)
    print(f"Saved: {output_filename}")



if __name__ == "__main__":
    root_folder = 'C:/Users/jessi/OneDrive/Dokument/ProjectCourse/data'

    for subdir, _, files in os.walk(root_folder):
        if 'raw' in subdir:  # Only look in 'raw' folders
            for file in files:
                if file.endswith('.pcd'):
                    # Read the point cloud
                    path = os.path.join(subdir, file)
                    pcd = o3d.io.read_point_cloud(path)

                    # Generate filename for saving
                    filename_folder = os.path.basename(os.path.dirname(subdir))  # Get the folder name
                    filename_file = os.path.splitext(file)[0]  # Get the file name without extension
                    filename = f"{filename_folder}_{filename_file}"
                    
                    # Split the chips into 4 parts
                    parts = split_chips(pcd)

                    for i, (part_points, part_colors) in enumerate(parts):
                        
                        # Remove empty chips
                        chip_present = remove_empty_chips(part_colors)

                        if chip_present:
                            part_points, part_colors = cut_chip_edges(part_points, part_colors)
                            chip = pc_from_numpy(part_points, part_colors)
                            
                            # Downsample chips
                            chip = downsample(chip)

                            # Remove noise
                            RANSAC_inliers, RANSAC_outliers = RANSAC(chip)
                            SOR_inliers, SOR_outliers = SOR(RANSAC_inliers)
                        
                            write_to_file(SOR_inliers, filename, i)