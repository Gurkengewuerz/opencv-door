from imutils.video import VideoStream
from flask import Flask, Response, render_template_string
import threading
import argparse
import pickle
import imutils
import time
import cv2

from variables import *

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(1.0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_names = []


@app.route("/")
def index():
    return render_template_string("""
<html>
  <head>
    <title>OpenCV Door</title>
    <style>
    div {
        height: 100%;
        position: relative;
    }

    img {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    </style>
  </head>
  <body style="background-color: #191919;">
    <div>
        <img src="{{ url_for('video_feed') }}">
    </div>
  </body>
</html>
    """)

def do_cv(needed_face_time = 3, access_time = 5, wait_time = 20):
    global vs, outputFrame, lock, face_cascade, recognizer, recognizer_names

    matchList = {}
    successStarted = 0
    facesIsZero = 0

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

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

            else:
                facesIsZero = facesIsZero + 1
                cv2.putText(frame, "No Match", (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if len(faces) == 0:
            facesIsZero = facesIsZero + 1
        
        if facesIsZero >= 500 // wait_time:
            matchList.clear()

        for val in matchList.values():
            t = time.time()
            if t - val["start"] > 5 and t - val["last"] < needed_face_time:
                successStarted = time.time()
                break
        
        if time.time() - successStarted < access_time:
            cv2.putText(frame, "ACCESS GRANTED", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 204, 0), thickness=2)
        else:
            successStarted = 0
        
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