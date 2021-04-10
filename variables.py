import os
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, "images")

cascade_dir = os.path.join(BASE_DIR, "data", "haarcascades")

face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))

trainings_dir = os.path.join(BASE_DIR, "trainings")

namePickle = os.path.join(trainings_dir, "names.pickle")

trainingFile = os.path.join(trainings_dir, "trainingData.yml")