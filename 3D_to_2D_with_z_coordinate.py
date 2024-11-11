import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import random 

RES = 0.00075
#RES = 0.00015 - #picture seems to be black, but its not, you have to zoom in

def scale_to_255(a, min_val, max_val, dtype=np.uint8):
    """Scales an array of values from a given min-max range to 0-255."""
    return (((a - min_val) / float(max_val - min_val)) * 255).astype(dtype)

def point_cloud_to_top_view(points, side_range=(-0.65, 0.65), fwd_range=(-0.42, 0.42), res=RES, saveto=None):
    """
    Creates a 2D top view of the point cloud with RGB values.
    
    Args:
        points: (numpy array) N x 6 array with x, y, z, r, g, b values for each point.
        side_range: (tuple) Range for the x-axis (left, right).
        fwd_range: (tuple) Range for the y-axis (back, front).
        res: (float) Resolution (in meters) per pixel.
        saveto: (str or None) Path to save the image.
    """
    # Extracts the x, y, z, and RGB values from the point cloud
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r_points = points[:, 3]
    g_points = points[:, 4]
    b_points = points[:, 5]

    # Calculate the center of the point cloud
    center_x = np.mean(x_points)
    center_y = np.mean(y_points)
    
    # Shift the point cloud so that the origin (0, 0, 0) is at the center of the image
    x_points -= center_x
    y_points -= center_y

    # Select only the points within the given ranges
    ff = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    ss = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))  
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    r_points = r_points[indices]
    g_points = g_points[indices]
    b_points = b_points[indices]

    # Use z-coordinate to modify color (example: use z for the blue channel)
    # Scale z to 0-255 range for color
    z_scaled = scale_to_255(z_points, min_val=np.min(z_points), max_val=np.max(z_points))

    # Scale the x, y values into pixel coordinates
    x_img = ((x_points - fwd_range[0]) / res).astype(np.int32)  # x → Pixel
    y_img = ((y_points - side_range[0]) / res).astype(np.int32)  # y → Pixel

    # Calculate image size based on resolution
    x_max = int((fwd_range[1] - fwd_range[0]) / res)
    y_max = int((side_range[1] - side_range[0]) / res)

    print(f"Image size: {x_max} pixels (width) x {y_max} pixels (height)")

    # Create an empty image
    im = np.zeros((y_max, x_max, 3), dtype=np.uint8)

    # Set the RGB values of the points as pixel values in the image
    # Use original colors for red and green channels, z-scaled values for blue channel
    im[y_img, x_img, 0] = scale_to_255(r_points, min_val=0, max_val=1)  # Red channel
    im[y_img, x_img, 1] = scale_to_255(g_points, min_val=0, max_val=1)  # Green channel
    im[y_img, x_img, 2] = z_scaled  # Blue channel based on z-coordinates

    # Save or display the image
    if saveto:
        Image.fromarray(im).save(saveto)
    else:
        plt.imshow(im)
        plt.axis('off')
        plt.show()

def load_pc_from_pcd(pcd_path):
    """Loads point cloud data from a PCD file using Open3D."""
    pcd = o3d.io.read_point_cloud(pcd_path)
    # Extract points and colors from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # Combine points and colors into an N x 6 array
    return np.hstack((points, colors))

random_number = random.randint(1, 99)

# Example call
points = load_pc_from_pcd('C:\\Users\\norad\\ETTEPLAN\\DATA\\chip_without_ears_scratches.ply') 
point_cloud_to_top_view(points, saveto=f'C:\\Users\\norad\\ETTEPLAN\\DATA\\TO_2D\\chip_without_ears_{random_number}_res{RES}_withZ.png')
