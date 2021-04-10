import cv2
import os
import numpy as np
import time
import argparse
from PIL import Image
import pickle

from variables import *

ap = argparse.ArgumentParser()
ap.add_argument("--capture", action='store_true', required=False,
                default=False, help="capture new images")
ap.add_argument("--train", action='store_true', required=False,
                default=False, help="train with images in the images folder")
args = vars(ap.parse_args())

if not args["capture"] and not args["train"]:
    ap.print_help()
    exit(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if args["capture"]:

    cap = cv2.VideoCapture(0)
    uname = input("Enter your name: ")

    uname_dir = os.path.join(image_dir, uname)

    if not os.path.exists(uname_dir):
        os.makedirs(uname_dir)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            cv2.putText(img, "No face found", (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        if len(faces) > 0:
            counter = 0
            for (x, y, w, h) in faces:
                gray_img = gray[y:y+h, x:x+w]
                path = os.path.join(
                    uname_dir, "{}-{}.jpg".format(int(time.time()), counter))
                print(path)
                cv2.imwrite(path, gray_img)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                counter = counter + 1
            cv2.waitKey(250)

        cv2.imshow("Preview: {}".format(uname), img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if args["train"]:
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []
    names = []

    total = 1
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                faceImg = Image.open(path).convert("L")
                faceNp = np.array(faceImg, "uint8")
                faces.append(faceNp)
                ids.append(total)
                names.append(label)
                total = total + 1

    recognizer.train(faces, np.array(ids))
    recognizer.save(trainingFile)

    with open(namePickle, "wb") as handle:
        pickle.dump(names, handle)
