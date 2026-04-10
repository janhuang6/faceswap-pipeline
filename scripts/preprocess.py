import os
import sys
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
import dlib
from skimage import transform as trans

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Face alignment reference points
REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953, 51.69630051],
    [65.53179625, 51.50139775],
    [48.02141115, 71.73660714],
    [33.54156287, 92.3655273],
    [62.37923232, 92.20410156]
], dtype=np.float32)


class FacePreprocessor:
    def __init__(self, input_dir, output_dir, source_person, img_size=256):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.source_person = source_person
        self.img_size = img_size

        self.source_dir = os.path.join(output_dir, 'source_faces')
        self.target_dir = os.path.join(output_dir, 'target_faces')

        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.target_dir, exist_ok=True)

        self.metadata = {
            'total_processed': 0,
            'source_count': 0,
            'target_count': 0,
            'failed_images': [],
            'skipped_images': [],
            'processed_persons': {}
        }

    def get_face_landmarks(self, image, face_rect):
        """Extract 68 facial landmarks from a face"""
        try:
            landmarks = predictor(image, face_rect)
            points = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)
            return points
        except Exception as e:
            return None

    def align_face(self, image, landmarks):
        """Align face using landmark points"""
        try:
            src_pts = np.float32([
                landmarks[36:42].mean(axis=0),  # Left eye
                landmarks[42:48].mean(axis=0),  # Right eye
                landmarks[30],  # Nose
                landmarks[48],  # Left mouth
                landmarks[54]  # Right mouth
            ])

            tform = trans.SimilarityTransform()
            tform.estimate(src_pts, REFERENCE_FACIAL_POINTS)
            aligned = trans.warp(image, tform.params, output_shape=(self.img_size, self.img_size))
            aligned = (aligned * 255).astype(np.uint8)

            return aligned
        except Exception as e:
            return None

    def detect_and_crop_face(self, image, face_rect):
        """Crop face region from image with padding"""
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        padding = int(0.2 * max(w, h))
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(image.shape[1], x + w + padding)
        y_max = min(image.shape[0], y + h + padding)

        cropped = image[y_min:y_max, x_min:x_max]
        return cropped

    def check_image_quality(self, image):
        """Check if image meets quality criteria"""
        if image.shape[0] < 100 or image.shape[1] < 100:
            return False, "Image too small"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return False, "Image too blurry"

        brightness = np.mean(gray)
        if brightness < 30 or brightness > 225:
            return False, "Image brightness out of range"

        return True, "OK"

    def normalize_image(self, image):
        """Normalize image brightness and contrast"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        normalized = cv2.merge([l_channel, a_channel, b_channel])
        normalized = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

        return normalized

    def process_image(self, img_path, person_name):
        """Process a single image"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                self.metadata['failed_images'].append({'path': img_path, 'reason': 'Could not read image'})
                return False

            faces = detector(image, 1)
            if len(faces) == 0:
                self.metadata['skipped_images'].append({'path': img_path, 'reason': 'No face detected'})
                return False

            face = max(faces, key=lambda f: f.width() * f.height())
            landmarks = self.get_face_landmarks(image, face)
            if landmarks is None:
                self.metadata['skipped_images'].append({'path': img_path, 'reason': 'Could not extract landmarks'})
                return False

            cropped_face = self.detect_and_crop_face(image, face)
            is_quality, quality_msg = self.check_image_quality(cropped_face)
            if not is_quality:
                self.metadata['skipped_images'].append(
                    {'path': img_path, 'reason': f'Quality check failed: {quality_msg}'})
                return False

            aligned_face = self.align_face(cropped_face, landmarks)
            if aligned_face is None:
                self.metadata['skipped_images'].append({'path': img_path, 'reason': 'Could not align face'})
                return False

            normalized_face = self.normalize_image(aligned_face)
            final_face = cv2.resize(normalized_face, (self.img_size, self.img_size))

            filename = os.path.splitext(os.path.basename(img_path))[0]

            if person_name == self.source_person:
                output_path = os.path.join(self.source_dir, f"{person_name}_{filename}.jpg")
                self.metadata['source_count'] += 1
            else:
                output_path = os.path.join(self.target_dir, f"{person_name}_{filename}.jpg")
                self.metadata['target_count'] += 1

            cv2.imwrite(output_path, final_face)
            self.metadata['total_processed'] += 1

            if person_name not in self.metadata['processed_persons']:
                self.metadata['processed_persons'][person_name] = 0
            self.metadata['processed_persons'][person_name] += 1

            return True

        except Exception as e:
            self.metadata['failed_images'].append({'path': img_path, 'reason': str(e)})
            return False

    def preprocess_dataset(self):
        """Scan all subfolders and preprocess images"""
        print(f"🔍 Scanning folder: {self.input_dir}")
        print(f"📊 Target image size: {self.img_size}x{self.img_size}\n")

        total_images = 0
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    total_images += 1

        print(f"📁 Found {total_images} images to process\n")

        processed = 0
        for root, dirs, files in os.walk(self.input_dir):
            person_name = os.path.basename(root)

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    processed += 1

                    if self.process_image(img_path, person_name):
                        status = "✅"
                    else:
                        status = "❌"

                    print(f"{status} [{processed}/{total_images}] {file[:30]:<30} - {person_name}")

        self._save_metadata()
        self._print_summary()

    def _save_metadata(self):
        metadata_path = os.path.join(self.output_dir, 'preprocessing_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"\n✅ Metadata saved: {metadata_path}")

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PREPROCESSING SUMMARY")
        print("=" * 70)
        print(f"✅ Total images processed: {self.metadata['total_processed']}")
        print(f"✅ Source faces (your face): {self.metadata['source_count']}")
        print(f"✅ Target faces (LFW): {self.metadata['target_count']}")
        print(f"⚠️  Skipped images: {len(self.metadata['skipped_images'])}")
        print(f"❌ Failed images: {len(self.metadata['failed_images'])}")
        print(f"📁 Person categories: {len(self.metadata['processed_persons'])}")
        print("\n📂 Output directories:")
        print(f"   Source: {self.source_dir}")
        print(f"   Target: {self.target_dir}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess LFW dataset with face detection and alignment')
    parser.add_argument('input_dir', type=str, help='Path to lfw_funneled folder')
    parser.add_argument('output_dir', type=str, help='Path to output folder')
    parser.add_argument('source_person', type=str, help='Name of source person folder')
    parser.add_argument('img_size', type=int, help='Image size to resize to')

    args = parser.parse_args()

    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        print("❌ Error: shape_predictor_68_face_landmarks.dat not found!")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)

    preprocessor = FacePreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source_person=args.source_person,
        img_size=args.img_size
    )

    preprocessor.preprocess_dataset()


if __name__ == '__main__':
    main()