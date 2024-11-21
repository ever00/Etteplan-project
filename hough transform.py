import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, draw, transform
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

inlier_cloud = o3d.io.read_point_cloud('C:/Users/jessi/OneDrive/Dokument/ProjectCourse/pipe_segmentations/2024-09-02_155419_243301-12_tray-b-4-a_L2_part_1_downsample_10_without_ears_seg_pipes.ply')
points = np.asarray(inlier_cloud.points)

projected_points = points[:, :2]

min_vals = np.min(projected_points, axis=0)
max_vals = np.max(projected_points, axis=0)

range_x = max_vals[0] - min_vals[0]
range_y = max_vals[1] - min_vals[1]

image_width = int(range_x * 1000) 
image_height = int(range_y * 1000)

image_size = (image_height, image_width)
image = np.zeros(image_size, dtype=np.uint8)

for point in projected_points:
    # Map coordinates to image space (normalize the points to the image size)
    x = int((point[0] - min_vals[0]) / range_x * (image_width - 1))  # Map to x-axis (horizontal)
    y = int((point[1] - min_vals[1]) / range_y * (image_height - 1))  # Map to y-axis (vertical)

    image[y, x] = 255

edges = feature.canny(image, sigma=2)

hough_radii = np.arange(26, 27, 1)  
hough_res = transform.hough_circle(edges, hough_radii)

accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, num_peaks=2000)

output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)


for center_x, center_y, radius in zip(cx, cy, radii):
    circy, circx = draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
    
    # Ensure the circle coordinates are within bounds of the image
    valid = (circx >= 0) & (circx < output_image.shape[1]) & (circy >= 0) & (circy < output_image.shape[0])
    circx, circy = circx[valid], circy[valid]
    
    # Assign the color (e.g., red) to each pixel on the circle perimeter
    output_image[circy, circx] = [255, 0, 0]

# Plot the original and result images side-by-side
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Original image (binary)
ax[0].imshow(edges, cmap='gray')
ax[0].set_title("Original Binary Image")
ax[0].axis('off')

# Image with detected circles highlighted
ax[1].imshow(output_image, cmap='gray')
ax[1].set_title("Detected Circles (Hough Transform)")
ax[1].axis('off')

plt.tight_layout()
plt.show()


# Back-project 2D circle detections to 3D
filtered_points = []
filtered_colors = []

# Back-project 2D circle detections to 3D
original_colors = np.asarray(inlier_cloud.colors)

for i, (x, y, r) in enumerate(zip(cx, cy, radii)):
    # Convert image coordinates back to 2D space
    x_world = x / (image_width - 1) * range_x + min_vals[0]
    y_world = y / (image_height - 1) * range_y + min_vals[1]
    radius_world = r / (image_width - 1) * range_x  # Assume uniform scaling

    # Find 3D points within the circle's projected radius
    distances = np.linalg.norm(projected_points - np.array([x_world, y_world]), axis=1)
    mask = distances <= radius_world
    circle_points = points[mask]
    circle_colors = original_colors[mask]

    # Compute the centroid of the circle
    centroid = np.mean(circle_points, axis=0)

    # Center the circle by subtracting the centroid
    centered_points = circle_points - centroid

    # Create a point cloud for this circle
    circle_cloud = o3d.geometry.PointCloud()
    circle_cloud.points = o3d.utility.Vector3dVector(centered_points)
    circle_cloud.colors = o3d.utility.Vector3dVector(circle_colors)

    # Save the circle as a separate file
    filename = f"2024-09-02_155419_243301-12_tray-b-4-a_L2_part_1_downsample_10_without_ears_seg_pipes{i+1}.ply"
    o3d.io.write_point_cloud(filename, circle_cloud)

    print(f"Saved circle {i+1} to {circle_cloud}")








