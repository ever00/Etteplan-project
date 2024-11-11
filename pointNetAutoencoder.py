# -*- coding: utf-8 -*-

!pip install open3d

!pip install plyfile

!unzip pipe_segmentations.zip

# Commented out IPython magic to ensure Python compatibility.
# %reset -f

import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.cluster import DBSCAN

def load_ply_files(folder_path, voxel_size=0.006):  # Adjust voxel_size to control downsampling
    ply_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('seg_pipes.ply'):
            filepath = os.path.join(folder_path, filename)
            ply = o3d.io.read_point_cloud(filepath)

            # Apply voxel downsampling to reduce point count
            downsampled_ply = ply.voxel_down_sample(voxel_size)
            ply_files.append(downsampled_ply)

            # Display the number of points after downsampling
            print(f"Number of points after downsampling '{filename}': {len(downsampled_ply.points)}")

    print(f"Loaded {len(ply_files)} PLY files with voxel downsampling.")
    return ply_files


def compute_average_points(ply_files):
    total_points = 0
    num_files = len(ply_files)

    for pcd in ply_files:
        total_points += np.asarray(pcd.points).shape[0]  # Count points in each PLY file

    average_points = total_points / num_files
    return average_points


def preprocess_point_cloud(pcd, num_points=27572):
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)

    if len(pcd_points) < num_points:
        # If there are fewer points, pad with zeros
        padded_points = np.zeros((num_points, 3), dtype=np.float32)
        padded_points[:len(pcd_points)] = pcd_points

        padded_colors = np.zeros((num_points, 3), dtype=np.float32)
        padded_colors[:len(pcd_colors)] = pcd_colors

        return padded_points, padded_colors
    else:
        # Sample points if there are enough
        sampled_indices = np.random.choice(len(pcd_points), num_points, replace=False)
        sampled_points = pcd_points[sampled_indices]
        sampled_colors = pcd_colors[sampled_indices]

        return sampled_points, sampled_colors


def chamfer_distance(p1, p2, batch_size=32):
    total_dist = 0
    num_points = p1.shape[1]
    num_batches = (num_points + batch_size - 1) // batch_size

    for i in range(0, num_points, batch_size):
        p1_batch = p1[:, i:i+batch_size, :]
        p2_batch = p2[:, i:i+batch_size, :]

        p1_coords, p1_colors = p1_batch[:, :, :3], p1_batch[:, :, 3:]
        p2_coords, p2_colors = p2_batch[:, :, :3], p2_batch[:, :, 3:]

        dist1_coords, _ = torch.min(torch.cdist(p1_coords, p2_coords), dim=1)
        dist2_coords, _ = torch.min(torch.cdist(p2_coords, p1_coords), dim=1)

        batch_dist = (
            torch.mean(dist1_coords) + torch.mean(dist2_coords)
        )

        total_dist += batch_dist

    normalized_dist = total_dist / num_batches
    return normalized_dist



class PointCloudDataset(Dataset):
    def __init__(self, ply_files, num_points=27572):
        self.ply_files = ply_files
        self.num_points = num_points

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        pcd = self.ply_files[idx]
        points, colors = preprocess_point_cloud(pcd, self.num_points)

        combined = np.concatenate((points, colors), axis=1)  # (num_points, num_features)
        return torch.tensor(combined, dtype=torch.float32)


class PointNetAutoencoder(nn.Module):
    def __init__(self):
        super(PointNetAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),  # Fixed to match encoder output size
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 27572 * 6),  # 6 features (3D + RGB)
        )

    def forward(self, x):
        # x shape (batch_size, num_points, num_features)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x.view(-1, 27572, 6)


import torch

def train_autoencoder(model, data_loader, optimizer, num_epochs=50, device='cpu'):
    model.train()
    all_coord_errors = []
    all_color_errors = []
    all_points = []
    accumulation_steps = 4
    scaler = torch.amp.GradScaler()  # For mixed precision training

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            data = data.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  # Use mixed precision
                reconstructed = model(data)  # Forward pass

                original_points = data[:, :, :3]
                reconstructed_points = reconstructed[:, :, :3]

                # Compute Chamfer distance loss
                loss = chamfer_distance(reconstructed_points, original_points)

            # Scale the loss and perform backpropagation
            scaler.scale(loss).backward()

            # Gradient accumulation steps
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            all_coord_errors.append(loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    return all_coord_errors

folder_path = '/content/pipe_segmentations'
ply_files = load_ply_files(folder_path)
average_points = compute_average_points(ply_files)

print(f"Average number of points: {average_points}")

# Create a dataset and dataloader
dataset = PointCloudDataset(ply_files)
data_loader = DataLoader(dataset, batch_size=12, pin_memory=True)

# Initialize model, optimizer
device = torch.device("cuda")
model = PointNetAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# Train the autoencoder
all_coord_errors = train_autoencoder(model, data_loader, optimizer, num_epochs=40, device=device)

def plot_point_cloud_3d(points, colors=None, title="Point Cloud", ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    colors = np.clip(colors, 0, 1)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=0.5, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def save_point_cloud_as_ply(points, colors, filename):
    colors = np.clip(colors, 0, 1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")


def plot_point_cloud_with_outliers(points, colors=None, coord_inliers=None, coord_outliers=None, color_inliers=None, color_outliers=None, title="Point Cloud", ax=None, save_filename=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        colors = np.clip(colors, 0, 1)

    if coord_inliers is not None:
        inlier_points = points[coord_inliers]  # Extract points for inliers
        inlier_colors = colors[coord_inliers] if colors is not None else None
        ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2],
                   c=inlier_colors, marker='o', s=0.5, alpha=0.7, label="Inliers")

    # Plot coordinate outliers in red
    if coord_outliers is not None:
        outlier_points = points[coord_outliers]
        outlier_colors = np.full((outlier_points.shape[0], 3), [1, 0, 0])
        ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2],
                   c=outlier_colors, marker='o', s=0.5, alpha=0.7, label="Coordinate Outliers")

    # Plot color outliers in blue
    if color_outliers is not None:
        outlier_points = points[color_outliers]
        outlier_colors = np.full((outlier_points.shape[0], 3), [0, 0, 1])
        ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2],
                   c=outlier_colors, marker='o', s=0.5, alpha=0.7, label="Color Outliers")

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()

    if save_filename is not None:
        save_point_cloud_as_ply(points, colors, save_filename)


def compute_errors(original_points, reconstructed_points, original_colors, reconstructed_colors):
    original_points = torch.tensor(original_points, dtype=torch.float32)
    reconstructed_points = torch.tensor(reconstructed_points, dtype=torch.float32)
    original_colors = torch.tensor(original_colors, dtype=torch.float32)
    reconstructed_colors = torch.tensor(reconstructed_colors, dtype=torch.float32)

    coord_errors = torch.norm(original_points - reconstructed_points, dim=1)
    color_errors = torch.norm(original_colors - reconstructed_colors, dim=1)
    return coord_errors, color_errors


def inliers_outliers(coord_errors, color_errors, coord_threshold, color_threshold, points):
    coord_inliers = (coord_errors <= coord_threshold).nonzero()[:, 0]  # Extract indices from the tensor
    coord_outliers = (coord_errors > coord_threshold).nonzero()[:, 0]
    color_inliers = (color_errors <= color_threshold).nonzero()[:, 0]
    color_outliers = (color_errors > color_threshold).nonzero()[:, 0]

    print(f"Number of coord inliers: {len(coord_inliers)}")
    print(f"Number of coord outliers: {len(coord_outliers)}")
    print(f"Number of color inliers: {len(color_inliers)}")
    print(f"Number of color outliers: {len(color_outliers)}")
    return coord_inliers, coord_outliers, color_inliers, color_outliers

data_iter = iter(data_loader)
original_data = next(data_iter).to(device)

model.eval()

with torch.no_grad():
    reconstructed_data = model(original_data)

original_points = original_data[0, :, :3].cpu().numpy()
original_colors = original_data[0, :, 3:].cpu().numpy()

reconstructed_points = reconstructed_data[0, :, :3].cpu().numpy()
reconstructed_colors = reconstructed_data[0, :, 3:].cpu().numpy()

print("Original points range:")
print(np.min(original_points, axis=0), np.max(original_points, axis=0))

print("Reconstructed points range:")
print(np.min(reconstructed_points, axis=0), np.max(reconstructed_points, axis=0))

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
plot_point_cloud_3d(original_points, colors=original_colors, title="Original Point Cloud", ax=ax1)

ax2 = fig.add_subplot(122, projection='3d')
plot_point_cloud_3d(reconstructed_points, colors=reconstructed_colors, title="Reconstructed Point Cloud", ax=ax2)

plt.show()

original_points = original_data[0, :, :3].cpu().numpy()
original_colors = original_data[0, :, 3:].cpu().numpy()

reconstructed_points = reconstructed_data[0, :, :3].cpu().numpy()
reconstructed_colors = reconstructed_data[0, :, 3:].cpu().numpy()

coord_errors, color_errors = compute_errors(original_points, reconstructed_points, original_colors, reconstructed_colors)

coord_threshold = 0.9
color_threshold = 1.87

coord_inliers, coord_outliers, color_inliers, color_outliers = inliers_outliers(coord_errors, color_errors, coord_threshold, color_threshold, original_points)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
plot_point_cloud_with_outliers(original_points, original_colors,
                               coord_inliers=coord_inliers, coord_outliers=coord_outliers,
                               color_inliers=color_inliers, color_outliers=color_outliers,
                               title="Original Point Cloud with Outliers", ax=ax)
plt.show()

def cluster_outliers(outliers, min_points=10, eps=0.5):
    db = DBSCAN(eps=eps, min_samples=min_points)
    labels = db.fit_predict(outliers)

    clusters = []
    for label in set(labels):
        if label != -1:  # Ignore noise points (label == -1)
            cluster_indices = np.where(labels == label)[0]
            clusters.append(cluster_indices)

    return clusters

def mark_grouped_outliers(position_clusters, min_cluster_size=5):
    grouped_outliers = set()

    for cluster in position_clusters:
        if len(cluster) > min_cluster_size:
            grouped_outliers.update(cluster)

    return np.array(list(grouped_outliers))

def plot_point_cloud_with_grouped_outliers(points, colors, coord_inliers=None, coord_outliers=None, grouped_outliers=None, color_inliers=[0, 0, 1], color_outliers=[1, 0, 0], title="Point Cloud", ax=None, save_filename=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    colors = np.clip(colors, 0, 1)

    coord_outliers = np.array(coord_outliers).astype(int) if coord_outliers is not None else None
    grouped_outliers = np.array(grouped_outliers).astype(int) if grouped_outliers is not None else None

    # Plot inliers
    if coord_inliers is not None:
        inlier_points = points[coord_inliers]
        inlier_colors = colors[coord_inliers] if colors is not None else np.full((inlier_points.shape[0], 3), color_inliers)
        ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2],
                   c=inlier_colors, marker='o', s=0.5, alpha=0.7, label="Inliers")

    # Plot grouped outliers
    if grouped_outliers is not None:
        grouped_outlier_points = points[grouped_outliers]
        grouped_outlier_colors = np.full((grouped_outlier_points.shape[0], 3), color_outliers)  # Red for grouped outliers
        ax.scatter(grouped_outlier_points[:, 0], grouped_outlier_points[:, 1], grouped_outlier_points[:, 2],
                   c=grouped_outlier_colors, marker='o', s=0.5, alpha=0.7, label="Grouped Position Outliers")

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Save to .ply file if filename provided
    if save_filename is not None:
        all_points = np.concatenate([inlier_points, grouped_outlier_points], axis=0)
        all_colors = np.concatenate([inlier_colors, grouped_outlier_colors], axis=0)
        save_point_cloud_as_ply(all_points, all_colors, save_filename)


position_outliers = original_points[coord_outliers]
position_clusters = cluster_outliers(position_outliers, min_points=10 , eps=0.03)
grouped_position_outliers = mark_grouped_outliers(position_clusters, min_cluster_size=1)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
plot_point_cloud_with_grouped_outliers(original_points, original_colors, coord_inliers=coord_inliers, coord_outliers=coord_outliers, grouped_outliers=grouped_position_outliers, title="Original Point Cloud with Grouped Outliers", ax=ax1, save_filename='True1.ply')
ax2 = fig.add_subplot(122, projection='3d')
plot_point_cloud_with_grouped_outliers(reconstructed_points, reconstructed_colors, coord_inliers=coord_inliers, coord_outliers=coord_outliers, grouped_outliers=grouped_position_outliers, title="Reconstructed Point Cloud with Grouped Outliers", ax=ax2, save_filename='True2.ply')
plt.show()

for chip_idx in range(5):
    # Extract original and reconstructed data for the current chip
    original_points = original_data[chip_idx, :, :3].cpu().numpy()
    original_colors = original_data[chip_idx, :, 3:].cpu().numpy()

    reconstructed_points = reconstructed_data[chip_idx, :, :3].cpu().numpy()
    reconstructed_colors = reconstructed_data[chip_idx, :, 3:].cpu().numpy()

    # Calculate errors for inliers and outliers
    coord_errors, color_errors = compute_errors(original_points, reconstructed_points, original_colors, reconstructed_colors)
    coord_inliers, coord_outliers, color_inliers, color_outliers = inliers_outliers(
        coord_errors, color_errors, coord_threshold, color_threshold, original_points
    )

    # Save original and reconstructed point clouds with inliers and outliers as .ply files
    original_filename = f"{chip_idx}_original_with_outliers.ply"
    reconstructed_filename = f"{chip_idx}_reconstructed_with_outliers.ply"

    # Prepare colors with inliers as white, coordinate outliers as red, color outliers as blue
    orig_colors_with_outliers = np.copy(original_colors)
    orig_colors_with_outliers[coord_outliers] = [1, 0, 0]  # Red for coord outliers
    orig_colors_with_outliers[color_outliers] = [0, 0, 1]  # Blue for color outliers

    # Prepare reconstructed colors similarly
    recon_colors_with_outliers = np.copy(reconstructed_colors)
    recon_colors_with_outliers[coord_outliers] = [1, 0, 0]  # Red for coord outliers
    recon_colors_with_outliers[color_outliers] = [0, 0, 1]  # Blue for color outliers

    # Save both original and reconstructed point clouds as .ply files
    save_point_cloud_as_ply(original_points, orig_colors_with_outliers, original_filename)
    save_point_cloud_as_ply(reconstructed_points, recon_colors_with_outliers, reconstructed_filename)

    print(f"Saved original and reconstructed point clouds for chip {chip_idx} as .ply files.")