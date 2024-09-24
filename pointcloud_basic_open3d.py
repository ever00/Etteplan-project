import open3d as o3d


# Visulize with open3d
pcd = o3d.io.read_point_cloud('C:/Users/jessi/OneDrive/Dokument/ProjectCourse/tray-b-4-a_L2.pcd')

# Downsample the pointcloud
voxel_size = 0.5
pcd_downsampled = pcd.voxel_down_sample(voxel_size)

# Get information about the pointcloud
print(pcd)

o3d.visualization.draw_geometries([pcd_downsampled], window_name="PCD Point Cloud", width=800, height=600)