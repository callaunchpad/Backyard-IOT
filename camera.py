import tensorflow.compat.v1 as tf
import keras
import numpy as np
import cv2
import h5py
import time
global sess, graph
sess = tf.InteractiveSession()
graph = tf.get_default_graph()
TFLITE = True
#changed kereas.models to tensorflow.keras.models for mobilenet
if not TFLITE:
  model = tf.keras.models.load_model('squeeze.h5', custom_objects = None, compile=True)
else:
  tflite_model = tf.lite.Interpreter('simple_fp16.tflite')
  tflite_model.allocate_tensors()
  inp_det = tflite_model.get_input_details()
  out_det = tflite_model.get_output_details()

#converter = tf.lite.TFLiteConverter.from_saved_model('mobile1.h5')
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#model = converter.convert()
print('model loaded')
if not TFLITE:
  model._make_predict_function()
print("next?")
font = cv2.FONT_HERSHEY_SIMPLEX

class Camera(object):
  def __init__(self):
    print('photo')
    self.capture = cv2.VideoCapture(0)

  def __del__(self):
    self.capture.release()

  def predict(self):
    global model, graph
    with sess.as_default():
      with graph.as_default():
        now = time.time()
        print('new loop', 0)
        ret, cap = self.capture.read()
        image = cv2.resize(cap, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        image = image/255
        image = image.astype(np.float32)
        now2 = time.time()
        print('resized', now2 - now)
        if not TFLITE:
            guess = model.predict(np.array(image))
        else:
            tflite_model.set_tensor(inp_det[0]['index'], image)
            tflite_model.invoke()
            guess = tflite_model.get_tensor(out_det[0]['index'])
        now3 = time.time()
        print('predicted', now3 - now2)
        convert = np.array(['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat',
          'cow', 'sheep', 'spider', 'squirrel'], dtype='<U8')
        bestGuess = max(guess[0])
        bestGuessIndex = np.where(guess[0] == bestGuess)[0][0]
        
        prediction = convert[bestGuessIndex]
        print(bestGuess)
        print(prediction, time.time() - now3)
        
        if bestGuess > .7:
          msg = 'confident: ' + str(bestGuess) + " " + prediction
          cv2.putText(cap, msg, (0, 50), font, 1, (255, 255, 0), 2)
        else:
          msg = "Not confident enough"
          msg2 = str(bestGuess) + " prob of " + prediction
          cv2.putText(cap, msg, (0, 50), font, 1, (255, 255, 0), 2)
          cv2.putText(cap, msg2, (0, 100), font, 1, (255, 255, 0), 2)
        
        _, jpeg = cv2.imencode('.jpg', cap)
        print('finished', time.time() - now)
        return jpeg.tobytes()
