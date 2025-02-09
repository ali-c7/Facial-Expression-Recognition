import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN

def detect_and_crop_face(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    

    detector = MTCNN()
    faces = detector.detect_faces(opencv_image)

    if len(faces) == 0:
        return image  # No face detected, return original image

    x, y, w, h = faces[0]['box']
    
    # Adding padding to include some background
    padding = 0.2
    x_pad = int(w * padding)
    y_pad = int(h * padding)

    x1 = max(x - x_pad, 0)
    y1 = max(y - y_pad, 0)
    x2 = min(x + w + x_pad, opencv_image.shape[1])
    y2 = min(y + h + y_pad, opencv_image.shape[0])

    face_image = opencv_image[y1:y2, x1:x2]
    
    # Convert back to PIL format
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_image_rgb)


def preprocess_image(image):
    face_image = detect_and_crop_face(image)
   
    if face_image.mode != 'L':
        face_image = face_image.convert('L')
    face_image = face_image.resize((48, 48))
    img_array = np.array(face_image)
    img_array = img_array.astype(np.float32)
   
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array, face_image