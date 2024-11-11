import os
import glob
import re
import shutil
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Base folder paths
base_folder = "/proj/uppmax2024-2-21/data/etteplan/data_original"
without_ears_dir = "/proj/uppmax2024-2-21/data/etteplan/data_downsampled10_without_ears0_01"
analysis_output_count = 0
ok_png_count = 0
not_ok_png_count = 0
scancodefail_count = 0  # Counter for files starting with "scancodefail"

# Target directories for labelled files
labelled_ok_dir = "/proj/uppmax2024-2-21/data/etteplan/labeled/OK"
labelled_not_ok_dir = "/proj/uppmax2024-2-21/data/etteplan/labeled/NOT_OK"

# Create target directories if they don't exist
os.makedirs(labelled_ok_dir, exist_ok=True)
os.makedirs(labelled_not_ok_dir, exist_ok=True)
labelled_ok_count = 0
labelled_not_ok_count = 0

# Dictionary to store extracted information
file_info = {}

# Traverse all subfolders in 'data_original'
for root, dirs, files in os.walk(base_folder):
    # Check if the current folder is 'analysis_output'
    if os.path.basename(root) == 'analysis_output':
        # Count this occurrence of 'analysis_output'
        analysis_output_count += 1
        
        # Count all .png files ending with OK.png
        ok_png_files = glob.glob(os.path.join(root, '*OK.png'))
        ok_png_count += len(ok_png_files)
        
        # Count all .png files that do not end with OK.png
        all_png_files = glob.glob(os.path.join(root, '*.png'))
        not_ok_png_count += len(all_png_files) - len(ok_png_files)
        
        # Search for a `raw` folder at the same level
        parent_folder = os.path.dirname(root)
        raw_folder = os.path.join(parent_folder, 'raw')
        parent_folder_date = os.path.basename(os.path.dirname(raw_folder))
        # Extract the two characters before _OK.png in each OK.png file name
        for file_path in ok_png_files:
            file_name = os.path.basename(file_path)
            #print(f"Checking file: {file_name}")
            
            # If the file name starts with 'scancodefail', ignore this file
            if file_name.lower().startswith('scancodefail'):
                scancodefail_count += 1
                continue  # Skip further processing for this file
            
            # Regex to extract the last two characters before _OK.png
            match = re.search(r"(.{2})_OK\.png$", file_name)
            if match:
                char = match.group(1)[0]   # First character
                part_index = match.group(1)[1]  # Second character
                #print(f"Match found for {file_name}: char={char}, part_index={part_index}")
                
                # Save 'char' and 'part_index' information in the dictionary
                file_info[file_name] = {'char': char, 'part_index': part_index, 'pcd_basefile': None}
                
                # If the `raw` folder exists, search for a matching .pcd file
                if os.path.exists(raw_folder):
                    #print(f"Raw folder does exist: {raw_folder}")
                    #print("Contents of raw folder:", os.listdir(raw_folder))
                    # Pattern for .pcd files containing the character `char` in the name
                    pcd_pattern = f"tray-b-4-{char.lower()}*.pcd"
                    #print(f"PCD pattern: {pcd_pattern}")
                    pcd_files = glob.glob(os.path.join(raw_folder, pcd_pattern))
                    #print("Found PCD files:", pcd_files)
                    # If a matching .pcd file is found, store its base name
                    if pcd_files:
                        # Only take the first matching .pcd file (if multiple found)
                        pcd_basefile = os.path.basename(pcd_files[0]).rsplit('.', 1)[0]
                        file_info[file_name]['pcd_basefile'] = pcd_basefile
                else:
                    print(f"Raw folder does not exist: {raw_folder}")

                # If pcd_basefile is still None, remove the entry
                if file_info[file_name]['pcd_basefile'] is None:
                    del file_info[file_name]
                    continue  

                # Pattern for .ply files, including the parent folder date
                ply_pattern = os.path.join(without_ears_dir, f"*{parent_folder_date}_{file_info[file_name]['pcd_basefile']}_part_{part_index}_*.ply")
                #print(f"Looking for .ply files with pattern: {ply_pattern}")
                ply_files = glob.glob(ply_pattern)

                
                # If a matching .ply file is found, store its path in the dictionary
                if ply_files:
                    file_info[file_name]['ply_file'] = ply_files[0]
                else:
                    #print(f"No matching ply file found for {file_name}")
                    del file_info[file_name]
                    continue  
            else:
                print(f"No match for {file_name}")

# # After the loop, check if file_info has been populated
# print(f"Number of entries in file_info: {len(file_info)}")
# if len(file_info) > 0:
#     print("Sample entries in file_info:")
#     for file_name, info in list(file_info.items())[:5]:  # Show first 5 entries if any
#         print(f"{file_name}: {info}")
# else:
#     print("No data found in file_info.")

# Summary messages
print(f"The 'analysis_output' folder was found {analysis_output_count} times.")
print(f"Number of files ending with OK.png: {ok_png_count}")
print(f"Number of files not ending with OK.png: {not_ok_png_count}")
print(f"Number of 'scancodefail' files: {scancodefail_count}")
# Print the number of entries in the dictionary
print(f"Number of entries in the dictionary: {len(file_info)}")

# Print extracted information on '_OK.png' files
# print("Extracted information on '_OK.png' files:")
# if file_info:
#     for file_name, info in file_info.items():
#         print(f"{file_name}: Char: {info['char']}, Part Index: {info['part_index']}, PCD Basefile: {info['pcd_basefile']}, PLY File: {info['ply_file']}")
# else:
#     print("No '_OK.png' files found or file names in the wrong format.")

#print('------------------')
############################# CHECK FOR DUPLICATES IN FILE INFO DICTIONARY

# Set to track already seen ply_file paths
seen_ply_files = set()

# List for duplicate entries
duplicates = []

# Check for duplicate .ply files in file_info
for file_name, info in file_info.items():
    ply_file_path = info.get('ply_file')  # Get the path of the .ply file
    
    if ply_file_path:
        # Check if the ply_file was already seen (duplicate)
        if ply_file_path in seen_ply_files:
            duplicates.append(file_name)  # Add to list of duplicates
        else:
            seen_ply_files.add(ply_file_path)  # Add to seen files
duplicate_count = len(duplicates)
print(f"Number of duplicate ply_file entries: {duplicate_count}")

# Output duplicates
if duplicates:
    print("Found duplicate ply_file entries:")
    for duplicate in duplicates:
        print(f"- {duplicate}")
else:
    print("No duplicate ply_file entries found.")

###################################### COPY

#### GO THROUGH FILE INFO DICTIONARY _ COPY ALL PLY FILES IN LABELED/OK

# # Copy .ply files listed in file_info to labelled_ok_dir
for file_name, info in file_info.items():
    ply_file_path = info.get('ply_file')  # Get the path of the .ply file

    #print(f"Processing file: {file_name}")
    #print(f"File path: {ply_file_path}")
    
    if ply_file_path and os.path.exists(ply_file_path):
        # Determine the destination path for the copy
        destination_path = os.path.join(labelled_ok_dir, os.path.basename(ply_file_path))
        
        try:
            # Copy the file to the destination folder
            shutil.copy(ply_file_path, destination_path)
            labelled_ok_count += 1
            #print(f"File {ply_file_path} would be copied to {destination_path}.")
        except Exception as e:
            print(f"Error copying the file '{ply_file_path}': {e}")
    else:
        print(f"The file '{ply_file_path}' does not exist or the path is invalid.")

print(f"Number of files in 'labelled_ok': {labelled_ok_count}")


#### GO THROUGH WITHOUT EARS _ COPY ALL FILES THAT ARE NOT IN LABELED/OK

# Copy .ply files in without_ears_dir that are not in labelled_ok_dir to labelled_not_ok_dir
for root, dirs, files in os.walk(without_ears_dir):
    for file in files:
        if file.endswith(".ply"):
            ply_file_path = os.path.join(root, file)

            # Skip files that already exist in 'labelled_ok'
            labelled_ok_file_path = os.path.join(labelled_ok_dir, file)
            if os.path.exists(labelled_ok_file_path):
                continue  # Skip to avoid printing when a file is found in labelled_ok_dir

            # Determine the destination path for the file in 'labelled_not_ok'
            destination_path = os.path.join(labelled_not_ok_dir, file)
            
            try:
                # Copy the file to 'labelled_not_ok'
                shutil.copy(ply_file_path, destination_path)
                labelled_not_ok_count += 1
            except Exception as e:
                print(f"Error copying the file '{ply_file_path}': {e}")

print(f"Number of files in 'labelled_not_ok': {labelled_not_ok_count}")
