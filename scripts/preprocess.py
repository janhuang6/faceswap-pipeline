import os
import cv2
import argparse
import json
import numpy as np

class FacePreprocessor:
    def __init__(self, input_dir, output_dir, source_person, img_size):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.source_person = source_person
        self.img_size = img_size
        self.metadata = {'processed_images': [], 'total_images': 0}

    def process_images(self):
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.metadata['total_images'] += 1
                    img_path = os.path.join(root, file)
                    self.process_image(img_path)

    def process_image(self, img_path):
        image = cv2.imread(img_path)
        # Here add face detection and preprocessing logic
        # For the sake of demonstration, let's assume the following dummy logic
        processed_image = cv2.resize(image, (self.img_size, self.img_size))
        output_path = os.path.join(self.output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, processed_image)
        
        self.metadata['processed_images'].append(output_path)

    def save_metadata(self):
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=4)
        print('Metadata saved!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory with images')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed images')
    parser.add_argument('--source-person', type=str, required=True, help='Name of the source person')
    parser.add_argument('--img-size', type=int, default=256, help='Size of the processed images')
    args = parser.parse_args()

    preprocessor = FacePreprocessor(args.input_dir, args.output_dir, args.source_person, args.img_size)
    preprocessor.process_images()
    preprocessor.save_metadata()