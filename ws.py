from imutils.video import VideoStream
from flask import Flask, Response, render_template_string, render_template
from gpiozero import LED
import threading
import requests
import argparse
import pickle
import imutils
import time
import cv2
import csv

from variables import *

led = LED(17)

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0).start()
time.sleep(1.0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_names = []


@app.route("/")
def index():
    return render_template('index.html')

def log(data):
    with open("log.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(int(time.time()))] + data)

def uploadImage(frame):
    print("Pushing image to pushover")
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
        "token": PUSHOVER_API_KEY,
        "user": PUSHOVER_USER_KEY,
        "message": "Unbekannter Zutritt"
    },
    files = {
        "attachment": ("image.jpg",  cv2.imencode('.jpg', frame)[1].tobytes(), "image/jpeg")
    })
    print(r.text)

def do_cv(needed_face_time = 3, access_time = 5, wait_time = 20):
    global vs, outputFrame, lock, face_cascade, recognizer, recognizer_names

    matchList = {}
    successStarted = 0
    facesIsZero = 0
    logged = False
    lastPush = 0

    while True:
        frame = vs.read()
        origFrame = frame
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        
        noMatch = False
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 80 and len(recognizer_names) > id_:
                facesIsZero = 0
                label = recognizer_names[id_]
                cv2.putText(frame, "{} {}".format(label, round(conf, 2)), (x+2, y+h-5), cv2.FONT_HERSHEY_DUPLEX, 1, (150, 255, 0))

                if label in matchList:
                    startedAt = matchList[label]["start"]
                    matchList[label] = {"last": time.time(), "start": startedAt}
                else:
                    matchList[label] = {"last": time.time(), "start": time.time()}
                    logged = False

            else:
                facesIsZero = facesIsZero + 1
                noMatch = True
                cv2.putText(frame, "No Match", (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if len(faces) == 0:
            facesIsZero = facesIsZero + 1
        
        if facesIsZero >= 500 // wait_time:
            matchList.clear()

        if noMatch and facesIsZero >= (1000 // wait_time) and time.time() - lastPush > 60:
            lastPush = time.time()
            log(["unknown", "RING"])
            thread = threading.Thread(target = uploadImage, args = (origFrame, ))
            thread.start()

        for val in matchList.values():
            t = time.time()
            if t - val["start"] > 5 and t - val["last"] < needed_face_time:
                successStarted = time.time()
                break
        
        if time.time() - successStarted < access_time:
            cv2.putText(frame, "ACCESS GRANTED", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 204, 0), thickness=2)
            led.on()
            if not logged:
                log([",".join(matchList.keys()), "GRANTED"])
                logged = True
        else:
            successStarted = 0
            led.off()
        
        with lock:
            outputFrame = frame.copy()

        cv2.waitKey(wait_time)



def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield(b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + 
            bytearray(encodedImage) + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    if not os.path.isfile(trainingFile):
        print("Please train the data first")
        exit(0)

    recognizer.read(trainingFile)
    with open(namePickle, "rb") as f:
        recognizer_names = pickle.load(f)

    t = threading.Thread(target=do_cv, args=())
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=6349, debug=True, threaded=True, use_reloader=False)

vs.stop()