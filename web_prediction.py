from flask import Flask, render_template, jsonify, Response
import requests
from time import sleep
from camera import Camera
import threading

app = Flask(__name__)
@app.route("/")
def index():
	return render_template("wildlifeIndex.html")
#global queue = []
global ret_frame
def predict(camera):
    global ret_frame
    while True:
        frame = camera.predict()
        ret_frame = frame
        #queue.appen(frame)


'''def gen_frame():
    yield from queue
'''

def generate():
    global ret_frame
    while True:
        frame = ret_frame #gen_frame()
        yield (b'--frame\r\n'+ b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/video_feed")
def image_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    camera = Camera()
    #x = threading.Thread(target=generate))
    y = threading.Thread(target=predict, args=(camera,))
    #x.start()
    y.start()
    app.run(host='0.0.0.0')
