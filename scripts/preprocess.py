import cv2
import numpy as np
import dlib
from retinaface import RetinaFace

# Function to detect faces
def detect_faces(image):
    faces = RetinaFace.detect_faces(image)
    return faces

# Function to extract facial landmarks
def extract_landmarks(image, rect):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    landmarks = predictor(image, rect)
    return landmarks

# Function to align faces
def align_face(face, landmarks, size=(256, 256)):
    # assuming landmarks[36:48] are the eyes
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    # Align and resize the face
    aligned_face = cv2.resize(face, size)
    return aligned_face

# Main processing function
def process_faces(image_path):
    image = cv2.imread(image_path)
    faces = detect_faces(image)
    processed_faces = []
    metadata = []

    for face in faces:
        rect = face['facial_area']
        landmarks = extract_landmarks(image, rect)
        aligned_face = align_face(image[rect[1]:rect[3], rect[0]:rect[2]], landmarks)
        processed_faces.append(aligned_face)
        metadata.append({'landmarks': landmarks})

    return processed_faces, metadata

# Example usage
if __name__ == '__main__':
    try:
        faces, meta = process_faces('path_to_image.jpg')
        print('Processed faces and metadata successfully.')
    except Exception as e:
        print(f'Error processing faces: {e}')