import tensorflow.compat.v1 as tf
import keras
from keras import applications
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

labels = np.load('data/labels.npy')
animals = np.load('data/label_to_animal.npz')
for i in animals:
    print(i)
animals = animals['arr_0']
animals.shape

images = np.load('data/images.npy')
images.shape

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)

base_model = load_model("base_model.h5")

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
predictions = keras.layers.Dense(16, activation= 'softmax')(x)
model = keras.models.Model(inputs = base_model.input, outputs = predictions)

from keras.optimizers import Adam
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
#changed loss to binary crossentropy from categorical_crossentropy to see what would happen

BATCH_SIZE=8

model.fit(X_train, y_train, epochs = 10, batch_size = BATCH_SIZE,
          validation_data=(X_test, y_test))

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print('test loss, test acc:', results)

model.save("model.h5")
