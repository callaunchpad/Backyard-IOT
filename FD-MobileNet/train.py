"""
 - Changed optimizer to SGD with nesterov momentum 0.9
 - Set init_lr to 0.01
 - Created step weight decay by 0.1 every 10 epochs
 - Added l2 regularization of 4e-5 to each layer in keras_fdmobilenet
 
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from keras_fdmobilenet import FDMobileNet
from collections import Counter

BATCH_SIZE=256
NUM_EPOCHS=30
INIT_LR=0.1
STEP=10
RATE=0.01

WIDTH, HEIGHT = (224, 224)
CLASSES=10
PATH = '../Data/animals10/raw-img/'

datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.1)

train_generator = datagen.flow_from_directory(
    PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='training')
#     rotation_range=10
#     width_shift_range=0.1
#     height_shift_range=0.1
#     zoom_range=0.1,

validation_generator = datagen.flow_from_directory(
    PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='validation')

counter = Counter(train_generator.classes)                        
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

#Multiplies INIT_LR by RATE every STEP epochs
scheduler = lambda epoch: INIT_LR * (epoch // STEP) * RATE
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = FDMobileNet()
    model.compile(optimizer=SGD(learning_rate=0.0, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights, callbacks=[callback]
)

model.save('model.h5')
