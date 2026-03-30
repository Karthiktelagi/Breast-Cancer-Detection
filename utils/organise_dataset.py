import os
import shutil

# Source dataset folder
source_base = "archive"

# Target folder
target_base = "data/train"

# Mapping BIRADS → classes
mapping = {
    "BIRAD1": "normal",
    "Birad3": "benign",
    "Birad4": "malignant",
    "Birad5": "malignant"
}

for birad_folder, label in mapping.items():
    src_path = os.path.join(source_base, birad_folder)

    # Check if folder exists
    if not os.path.exists(src_path):
        print("Folder not found:", src_path)
        continue

    # Get subfolders (b1, b3, b4, etc.)
    subfolders = os.listdir(src_path)

    # Go inside subfolder if exists
    if len(subfolders) > 0:
        src_path = os.path.join(src_path, subfolders[0])

    # Read images
    for file in os.listdir(src_path):
        src_file = os.path.join(src_path, file)

        # Skip if not a file
        if not os.path.isfile(src_file):
            continue

        # Create destination folder
        dst_folder = os.path.join(target_base, label)
        os.makedirs(dst_folder, exist_ok=True)

        # Copy image
        dst_file = os.path.join(dst_folder, file)
        shutil.copy(src_file, dst_file)

print("✅ Dataset organized successfully!")