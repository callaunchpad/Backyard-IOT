from flask import Flask, render_template, jsonify, Response
import requests
from time import sleep
from camera import Camera

app = Flask(__name__)
@app.route("/")
def index():
	return render_template("wildlifeIndex.html")

def generate(camera):
	while True:
		frame = camera.predict()
		yield (b'--frame\r\n'
	               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/video_feed")
def image_feed():
	return Response(generate(Camera()),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
	app.run(host='0.0.0.0')
