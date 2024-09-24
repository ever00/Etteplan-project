import subprocess

# Full path to the CloudCompare executable
cloudcompare_path = r'C:\Program Files\CloudCompare\CloudCompare.exe'

# Path to the PCD file
pcd_file_path = r'C:\Users\jessi\OneDrive\Dokument\ProjectCourse\tray-b-4-a_L2.pcd'

# Command to open CloudCompare with the PCD file
command = rf'"{cloudcompare_path}" -CLEAR -AUTO_SAVE OFF -o "{pcd_file_path}" -SHOW'

# Print command for debugging
print(f"Running command: {command}")

# Run the command
subprocess.run(command, shell=True)