from flask import Flask, render_template, jsonify, Response
import requests
from time import sleep
import requests
from jsonCamera import Camera

counter = 0
totalTime = 0
camera = Camera()

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("jsonIndex.html")

# def generate(camera):
#     counter = 0
#     total = 0
#     while True:
#         frame, newTime = camera.predict()
#         if counter > 9:
#             total += newTime
#         counter += 1
#         yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         if counter == 60:
#             print("STOP STOP STOP STOP STOP \n")
#             print(total/50)
#             break
# 
# @app.route("/video_feed")
# def image_feed():
#     return Response(generate(Camera()),
#         mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/animals", methods=['GET'])
def get_results():
    global totalTime, counter
    res, time = camera.predict()
    if counter > 9:
        totalTime += time
    counter += 1
    if counter == 60:
        print("=====================" + "\n" + str(totalTime/50))
    
    return jsonify(res)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
