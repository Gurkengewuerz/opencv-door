from imutils.video import VideoStream
from flask import Flask, Response, render_template_string, render_template
from gpiozero import LED
import threading
import requests
import argparse
import pickle
import imutils
import base64
import time
import cv2
import csv

from variables import *

led = LED(17)

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
vs = VideoStream(src=0).start()
time.sleep(1.0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_names = []

last_refresh_permissions = 0
permissions = []
lastPush = {"no_match": 0}
lastLog = {"no_match": 0}


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/log")
def log_page():
    with open("log.csv", "a", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        return render_template('log.html', log_entries=reader)

@app.route("/permissions")
def permissions_page():
    return render_template('permissions.html', permissions=permissions)

def log(data):
    with open("log.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(int(time.time()))] + data)

def uploadImage(frame):
    if PUSHOVER_API_KEY == "" or PUSHOVER_USER_KEY == "":
        print("Skipping pushover")
        return
    print("Pushing image to pushover")
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
        "token": PUSHOVER_API_KEY,
        "user": PUSHOVER_USER_KEY,
        "message": "Unbekannter Zutritt"
    },
    files = {
        "attachment": ("image.jpg",  frame, "image/jpeg")
    })
    print(r.text)

def push(label, frame):
    lastPush[label] = time.time()
    thread = threading.Thread(target = uploadImage, args = (frame, ))
    thread.start()

def do_cv(needed_face_time = 3, access_time = 5, wait_time = 20):
    global vs, outputFrame, lock, face_cascade, recognizer, recognizer_names, last_refresh_permissions, lastLog, lastPush, permissions

    matchList = {}
    successStarted = 0
    facesIsZero = 0
    logged = False

    while True:
        if time.time() - last_refresh_permissions > 60:
            last_refresh_permissions = time.time()
            with open("permissions.csv", "r", newline='') as csvfile:
                perms = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                permissions.clear()
                for row in perms:
                    permissions.append(row)

        frame = vs.read()
        origFrame = frame
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        
        noMatch = False
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50 and len(recognizer_names) > id_:
                facesIsZero = 0
                label = recognizer_names[id_]
                cv2.putText(frame, "{} {}".format(label, round(conf, 2)), (x+2, y+h-5), cv2.FONT_HERSHEY_DUPLEX, 1, (150, 255, 0))

                if label in matchList:
                    startedAt = matchList[label]["start"]
                    known_permission = matchList[label]["known_permission"]
                    matchList[label] = {"last": time.time(), "start": startedAt, "known_permission": known_permission, "name": label}
                else:
                    known_permission = []
                    for entry in permissions:
                        if entry[0].lower() == label.lower():
                            known_permission = entry[1].lower().split(",")

                    matchList[label] = {"last": time.time(), "start": time.time(), "known_permission": known_permission, "name": label}
                    if label not in lastPush:
                        lastPush[label] = 0
                    if label not in lastLog:
                        lastLog[label] = 0

            else:
                facesIsZero = facesIsZero + 1
                noMatch = True
                logged = False
                cv2.putText(frame, "No Match", (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if len(faces) == 0:
            facesIsZero = facesIsZero + 1
        
        if facesIsZero >= 500 // wait_time:
            matchList.clear()

        if noMatch and facesIsZero >= (1000 // wait_time) and time.time() - lastPush["no_match"] > 60:
            imgBytes = cv2.imencode('.jpg', frame)[1].tobytes()
            push("no_match", imgBytes)
            if time.time() - lastLog["no_match"] > 30:
                lastLog["no_match"] = time.time()
                log(["unknown", "RING", base64.b64encode(imgBytes)])

        for val in matchList.values():
            t = time.time()
            if t - lastPush[val["name"]] > 60 and "push" in val["known_permission"]:
                push(val["name"], cv2.imencode('.jpg', frame)[1].tobytes())
            if t - lastLog[val["name"]] > 60 and "log" in val["known_permission"]:
                lastLog[val["name"]] = time.time()
                log([val["name"], "SEEN"])
            if t - val["start"] > 5 and t - val["last"] < needed_face_time and "access" in val["known_permission"]:
                successStarted = time.time()
                break
        
        if time.time() - successStarted < access_time:
            cv2.putText(frame, "ACCESS GRANTED", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 204, 0), thickness=2)
            if not logged:
                logged = True
                led.on()
                log([",".join(matchList.keys()), "GRANTED"])
        else:
            logged = False
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