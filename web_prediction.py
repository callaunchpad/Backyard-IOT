from flask import Flask, render_template, jsonify, Response
import requests
from time import sleep
from jsonCamera import Camera
import threading

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("jsonIndex.html")
#global queue = []
global ret_frame
def predict(camera):
    global ret_frame
    global time
    while True:
        frame, time = camera.predict()
        ret_frame = frame


'''def gen_frame():
    yield from queue
'''

# def generate():
#     global ret_frame
#     counter = 0
#     total = 0
#     while True:
#         frame = ret_frame #gen_frame()
#         newTime = time
#         if counter > 49:
#             total += newTime
#         counter += 1
#         #yield (b'--frame\r\n'+ b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         frame = str.encode(frame)
#         yield(frame)
#         if counter == 100:
#             print("==============================" + "\n")
#             print(total/50)
#             print("==============================" + "\n")

@app.route("/animals")
def get_results():
    return jsonify(ret_frame)

if __name__ == "__main__":
    camera = Camera()
    #x = threading.Thread(target=generate))
    y = threading.Thread(target=predict, args=(camera,))
    #x.start()
    y.start()
    app.run(host='0.0.0.0')
