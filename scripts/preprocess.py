import os
import cv2
import json
import tqdm

# Constants
SOURCE_DIR = 'lfw_funneled/'
TARGET_DIR = 'processed_faces/'
IMAGE_SIZE = (256, 256)
META_DATA_FILE = 'metadata.json'

# Ensure the target directory exists
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# Metadata storage
metadata = {'source_faces': [], 'target_faces': []}

def process_face(image_path, is_source):
    try:
        # Read the image
        image = cv2.imread(image_path)
        # Resize the image
        image_resized = cv2.resize(image, IMAGE_SIZE)
        # Create target path
        face_name = os.path.basename(image_path)
        target_path = os.path.join(TARGET_DIR, face_name)
        # Save the processed image
        cv2.imwrite(target_path, image_resized)
        # Append metadata
        if is_source:
            metadata['source_faces'].append(target_path)
        else:
            metadata['target_faces'].append(target_path)
    except Exception as e:
        print(f'Error processing {image_path}: {e}')  # Error handling

# Counting total images for progress tracking
total_files = sum(len(files) for _, _, files in os.walk(SOURCE_DIR))

for root, dirs, files in os.walk(SOURCE_DIR):
    for file in tqdm.tqdm(files, desc='Processing images', total=total_files):
        file_path = os.path.join(root, file)
        # Check if it's an image file (you can modify this check based on requirements)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Assume source faces are in 'source' folder and target faces are in 'target' folder
            is_source = 'source' in root
            process_face(file_path, is_source)

# Save the metadata to a JSON file
with open(META_DATA_FILE, 'w') as meta_file:
    json.dump(metadata, meta_file, indent=4)
