from flask import Flask, render_template, jsonify
import requests
from time import sleep
app = Flask(__name__)
i = 1

@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/animals", methods=['GET'])
def get_animals():
    global i 
    l = ["cat", "dog", "pikachu"]
    st = l[i%len(l)]
    i+=1

    return jsonify(st)

if __name__ == "__main__":
    app.run(host='0.0.0.0')