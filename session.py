import tensorflow as tf
import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import numpy as np
from sklearn.model_selection import train_test_split

strategy = tf.distribute.MirroredStrategy()

images = np.load('images.npy')
labels = np.load('labels.npy')
print(images.shape)
INPUT_SHAPE = images[0].shape

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
del images
del labels

y_lst = list(np.argmax(y_train, axis=1))
unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
class_weights = {unique[i]:counts[i] for i in range(len(unique))}
for i in range(NUM_CLASSES):
    if i not in class_weights:
        class_weights[i] = 0
        
base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=INPUT_SHAPE)

DROPOUT = 0.2
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(DROPOUT)(x)
predictions = keras.layers.Dense(NUM_CLASSES, activation= 'softmax')(x)

from keras.optimizers import Adam

adam = Adam(lr=0.005)

with strategy.scope():
    model = keras.models.Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
NUM_EPOCHS = 100
BATCH_SIZE = 10

model.fit(X_train, y_train, epochs = NUM_EPOCHS, batch_size = BATCH_SIZE,
          validation_data=(X_test, y_test), class_weight=class_weights)

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print('test loss, test acc:', results)

model.save("model.h5")
