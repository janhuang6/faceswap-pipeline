import os
import cv2
import dlib
from tqdm import tqdm

DATA_DIR = ".\\data\\raw\\lfw_funneled"
ALIGNED_DIR = ".\\data\\aligned"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download

os.makedirs(ALIGNED_DIR, exist_ok=True)

for subdir_name in tqdm(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, subdir_name)
    print("dir_path: ", dir_path)
    if os.path.isdir(dir_path):
        for img_name in tqdm(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            print("Now processing: ", img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) == 0:
                continue
            face = faces[0]
            aligned_face = img[face.top():face.bottom(), face.left():face.right()]
            aligned_face = cv2.resize(aligned_face, (128, 128))  # start small
            cv2.imwrite(os.path.join(ALIGNED_DIR, img_name), aligned_face)