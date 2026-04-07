import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

class DatasetUpdater:
    def __init__(self, base_dir='data', source_face_dir='source_faces', target_face_dir='target_faces'):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.source_dir = self.processed_dir / source_face_dir
        self.target_dir = self.processed_dir / target_face_dir
        self.metadata_file = self.base_dir / 'dataset_metadata.json'
        self._create_directories()
        self._load_metadata()
    
    def _create_directories(self):
        for directory in [self.raw_dir, self.source_dir, self.target_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'total_images': 0,
                'source_count': 0,
                'target_count': 0,
                'last_update': None,
                'images': {}
            }
    
    def _save_metadata(self):
        self.metadata['last_update'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_source_face(self, image_path, destination_name=None):
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return False
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return False
        dest_name = destination_name or image_path.stem
        dest_path = self.source_dir / f"{dest_name}.jpg"
        cv2.imwrite(str(dest_path), img)
        print(f"✅ Added source face: {dest_path}")
        self.metadata['images'][str(dest_path)] = {'type': 'source', 'added': datetime.now().isoformat()}
        self.metadata['source_count'] += 1
        self.metadata['total_images'] += 1
        self._save_metadata()
        return True
    
    def add_target_faces(self, source_dir, label=None):
        source_dir = Path(source_dir)
        if not source_dir.exists():
            print(f"❌ Directory not found: {source_dir}")
            return 0
        count = 0
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        print(f"📂 Scanning directory: {source_dir}")
        for image_path in tqdm(source_dir.rglob('*')):
            if image_path.suffix.lower() in valid_extensions:
                img = cv2.imread(str(image_path))
                if img is not None:
                    relative_path = image_path.relative_to(source_dir)
                    dest_path = self.target_dir / relative_path.name
                    cv2.imwrite(str(dest_path), img)
                    self.metadata['images'][str(dest_path)] = {'type': 'target', 'label': label, 'added': datetime.now().isoformat()}
                    count += 1
        self.metadata['target_count'] += count
        self.metadata['total_images'] += count
        self._save_metadata()
        print(f"✅ Added {count} target faces from {source_dir}")
        return count
    
    def organize_lfwpeople_dataset(self, lfwpeople_path):
        lfwpeople_path = Path(lfwpeople_path)
        if not lfwpeople_path.exists():
            print(f"❌ LFW People directory not found: {lfwpeople_path}")
            return 0
        print("📥 Organizing LFW People dataset...")
        count = self.add_target_faces(str(lfwpeople_path), label='lfw_people')
        return count
    
    def get_dataset_stats(self):
        stats = {'total_images': self.metadata['total_images'], 'source_faces': self.metadata['source_count'], 'target_faces': self.metadata['target_count'], 'last_update': self.metadata['last_update'], 'source_dir': str(self.source_dir), 'target_dir': str(self.target_dir)}
        return stats
    
    def print_stats(self):
        stats = self.get_dataset_stats()
        print("\n" + "="*50)
        print("📊 DATASET STATISTICS")
        print("="*50)
        print(f"Total Images: {stats['total_images']}")
        print(f"Source Faces: {stats['source_faces']}")
        print(f"Target Faces: {stats['target_faces']}")
        print(f"Last Update: {stats['last_update']}")
        print(f"Source Dir: {stats['source_dir']}")
        print(f"Target Dir: {stats['target_dir']}")
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Update and manage face swap dataset')
    parser.add_argument('--add-source', type=str, help='Path to source face image')
    parser.add_argument('--add-targets', type=str, help='Path to directory with target faces')
    parser.add_argument('--organize-lfw', type=str, help='Path to LFW People dataset')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    parser.add_argument('--base-dir', type=str, default='data', help='Base data directory')
    args = parser.parse_args()
    updater = DatasetUpdater(base_dir=args.base_dir)
    if args.add_source:
        updater.add_source_face(args.add_source)
    if args.add_targets:
        updater.add_target_faces(args.add_targets)
    if args.organize_lfw:
        updater.organize_lfwpeople_dataset(args.organize_lfw)
    if args.stats or not (args.add_source or args.add_targets or args.organize_lfw):
        updater.print_stats()

if __name__ == '__main__':
    main()