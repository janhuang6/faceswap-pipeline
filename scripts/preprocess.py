import os
import argparse
import cv2
import json
from tqdm import tqdm


def preprocess_images(input_dir, output_dir, source_person, img_size):
    source_faces_dir = os.path.join(output_dir, 'source_faces')
    target_faces_dir = os.path.join(output_dir, 'target_faces')

    os.makedirs(source_faces_dir, exist_ok=True)
    os.makedirs(target_faces_dir, exist_ok=True)

    metadata = {'source_faces': [], 'target_faces': []}

    # Scan through the input directory and subfolders
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc='Processing images'):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    # Read the image
                    image = cv2.imread(img_path)
                    if image is None:
                        raise ValueError('Image could not be read.')

                    # Resize the image
                    image = cv2.resize(image, (img_size, img_size))

                    if source_person in root:
                        dest_path = os.path.join(source_faces_dir, file)
                        metadata['source_faces'].append(dest_path)
                    else:
                        dest_path = os.path.join(target_faces_dir, file)
                        metadata['target_faces'].append(dest_path)

                    # Save the image
                    cv2.imwrite(dest_path, image)
                except Exception as e:
                    print(f'Error processing {img_path}: {e}')

                    # Save metadata to JSON
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as json_file:
        json.dump(metadata, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess images for face swap pipeline.')
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('output_dir', type=str, help='Output directory for processed images')
    parser.add_argument('source_person', type=str, help='Name of the source person')
    parser.add_argument('img_size', type=int, help='Size to resize images')
    args = parser.parse_args()

    preprocess_images(args.input_dir, args.output_dir, args.source_person, args.img_size)
