
import cv2
import pickle
import os
import time

face_cascade = cv2.CascadeClassifier(
    'data/haarcascades/haarcascade_frontalface_default.xml')

fname = "trainings/trainingData.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)

names = []
with open("trainings/names.pickle", "rb") as f:
    names = pickle.load(f)

cap = cv2.VideoCapture(0)

matchList = {}
successStarted = 0

NEEDED_FACE_TIME = 3
ACCESS_TIME = 5
WAIT_TIME = 20

facesIsZero = 0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 80 and len(names) > id_:
            facesIsZero = 0
            label = names[id_]
            cv2.putText(frame, "{} {}".format(label, round(conf, 2)), (x+2, y+h-5), cv2.FONT_HERSHEY_DUPLEX, 1, (150, 255, 0))

            if label in matchList:
                startedAt = matchList[label]["start"]
                matchList[label] = {"last": time.time(), "start": startedAt}
            else:
                matchList[label] = {"last": time.time(), "start": time.time()}

        else:
            facesIsZero = facesIsZero + 1
            cv2.putText(frame, 'No Match', (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if len(faces) == 0:
        facesIsZero = facesIsZero + 1
    
    if facesIsZero >= 500 // WAIT_TIME:
        matchList.clear()

    for val in matchList.values():
        t = time.time()
        if t - val["start"] > 5 and t - val["last"] < 2:
            successStarted = time.time()
            break
    
    if time.time() - successStarted < ACCESS_TIME:
        cv2.putText(frame, "ACCESS GRANTED", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 204, 0), thickness=2)
    else:
        successStarted = 0

    cv2.imshow("Preview", frame)
    if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
