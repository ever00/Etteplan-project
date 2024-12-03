import open3d as o3d
import numpy as np
from skimage import feature, draw, transform
import os
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Parameters
MIN_DISTANCE = 70  # Minimum distance between circle centers
MAX_PEAKS = 128    # Maximum number of circles to process per file
RADIUS_SCALING_FACTOR = 1000  # Scaling for the Hough radius

# Load PLY files
def load_ply_files(folder_path):
    ply_files = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('seg_pipes.ply'):
            filepath = os.path.join(folder_path, filename)
            ply = o3d.io.read_point_cloud(filepath)
            ply_files.append(ply)
            filenames.append(filename)
    print(f"Loaded {len(ply_files)} PLY files.")
    return ply_files, filenames

# Folder paths
folder_path = 'C:/Users/jessi/OneDrive/Dokument/ProjectCourse/data_downsampled0_without_ears0_01_segmented/Pipes'
output_dir = "non_downsampeled_hough"
os.makedirs(output_dir, exist_ok=True)

# Load files
ply_files, filenames = load_ply_files(folder_path)

# Process each PLY file
# Iterate over PLY files and process circles
for inlier_cloud, filename in zip(ply_files, filenames):
    points = np.asarray(inlier_cloud.points)
    original_colors = np.asarray(inlier_cloud.colors)

    # 2D Projection
    projected_points = points[:, :2]
    min_vals = np.min(projected_points, axis=0)
    max_vals = np.max(projected_points, axis=0)
    range_x = max_vals[0] - min_vals[0]
    range_y = max_vals[1] - min_vals[1]

    # Create image
    image_width = int(range_x * RADIUS_SCALING_FACTOR)
    image_height = int(range_y * RADIUS_SCALING_FACTOR)
    image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Normalize and project points into 2D space
    for point in projected_points:
        x = int((point[0] - min_vals[0]) / range_x * (image_width - 1))
        y = int((point[1] - min_vals[1]) / range_y * (image_height - 1))
        image[y, x] = 255

    # Edge detection
    edges = feature.canny(image, sigma=1.3, low_threshold=185, high_threshold=200)

    # Hough Circle Transform
    hough_radii = np.array([27])  # Adjust radii as needed
    hough_res = transform.hough_circle(edges, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii)

    # Collect detected circles
    circles = sorted(zip(accums, cx, cy, radii), reverse=True, key=lambda x: x[0])

    # Filter circles based on minimum distance and limit the number of detections
    final_centers = []
    final_radii = []
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for acc, center_x, center_y, radius in circles:
        if len(final_centers) >= MAX_PEAKS:
            break  # Stop if the maximum number of circles is reached

        overlap = any(
            distance((center_x, center_y), (fc[0], fc[1])) < MIN_DISTANCE
            for fc in final_centers
        )
        if not overlap:
            final_centers.append((center_x, center_y))
            final_radii.append(radius)
            circy, circx = draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
            # Ensure the circle coordinates are within bounds of the image
            valid = (circx >= 0) & (circx < output_image.shape[1]) & (circy >= 0) & (circy < output_image.shape[0])
            circx, circy = circx[valid], circy[valid]
            
            # Assign the color (e.g., red) to each pixel on the circle perimeter
            output_image[circy, circx] = [255, 0, 0]

    # Debug: Check detected circles
    print(f"{filename}: {len(final_centers)} circles detected after filtering.")


    # Sort circles first by x-coordinate, then by y-coordinate for left-to-right, top-to-bottom order
    final_centers_sorted = sorted(zip(final_centers, final_radii), key=lambda x: (x[0][0], x[0][1]))

    # Initialize processed_circles for this file
    processed_circles = set()

    # Back-project and save each circle
    for i, ((center_x, center_y), r) in enumerate(final_centers_sorted):
        x_world = center_x / (image_width - 1) * range_x + min_vals[0]
        y_world = center_y / (image_height - 1) * range_y + min_vals[1]
        radius_world = r / (image_width - 1) * range_x  # Assume uniform scaling

        distances = np.linalg.norm(projected_points - np.array([x_world, y_world]), axis=1)
        mask = distances <= radius_world
        circle_points = points[mask]
        circle_colors = original_colors[mask]

        if len(circle_points) < 10:
            continue  # Skip if not enough points

        centroid = np.mean(circle_points, axis=0)
        centered_points = circle_points - centroid

        circle_key = (round(x_world, 3), round(y_world, 3), round(radius_world, 3))
        if circle_key in processed_circles:
            continue
        processed_circles.add(circle_key)

        circle_cloud = o3d.geometry.PointCloud()
        circle_cloud.points = o3d.utility.Vector3dVector(centered_points)
        circle_cloud.colors = o3d.utility.Vector3dVector(circle_colors)

        # Generate a unique filename for each circle with left-to-right, top-to-bottom numbering
        circle_filename = f"{os.path.splitext(filename)[0]}_circle_{i+1}.ply"
        circle_filepath = os.path.join(output_dir, circle_filename)
        o3d.io.write_point_cloud(circle_filepath, circle_cloud)

    print(f"Completed processing for file: {filename}")