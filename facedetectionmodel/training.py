import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    facesSample = []
    ids = []

    for imagepath in imagepaths:
        PILimage = Image.open(imagepath).convert('L')
        img_numpy = np.array(PILimage, 'uint8')

        filename = os.path.split(imagepath)[-1]
        id_str = filename.split(".")[1].split("-")[0]

        try:
           id = int(id_str)
        except ValueError:
           print(f"Error: Unable to convert '{id_str}' to an integer.")
    # Handle the error or skip this image

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            facesSample.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
            
    return facesSample, ids

print("\n [INFO] Training faces ......")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')

print('\n [info] {0} faces trained.'.format(len(np.unique(ids))))


         
    
    
    