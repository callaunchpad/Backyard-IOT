import tensorflow.compat.v1 as tf
import keras
import numpy as np
import cv2
import h5py
import time

global sess, graph
sess = tf.InteractiveSession()
graph = tf.get_default_graph()

file = h5py.File('model.h5', 'r')
print('file loaded')
model = keras.models.load_model(file, custom_objects = None, compile=True)
print('model loaded')
model._make_predict_function()

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
				image = cv2.resize(cap, (1000, 500), interpolation = cv2.INTER_AREA)
				image = np.expand_dims(image, axis=0)

				now2 = time.time()
				print('resized', now2 - now)
				guess = model.predict(np.array(image))

				now3 = time.time()
				print('predicted', now3 - now2)
				convert = np.array(['bobcat', 'opossum', 'empty', 'coyote', 'raccoon', 'bird', 'dog',
		        	       'cat', 'squirrel', 'rabbit', 'skunk', 'rodent', 'badger', 'deer',
	        		       'car', 'fox'], dtype='<U8')
				bestGuess = max(guess[0])
				bestGuessIndex = np.where(guess[0] == bestGuess)[0][0]

				prediction = convert[bestGuessIndex]
				print(bestGuess)
				print(prediction, time.time() - now3)

				if bestGuess > .6:
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
