"""
 - removed dropout
 - decreased learning rate by 1/10
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras_fdmobilenet import FDMobileNet
from collections import Counter

BATCH_SIZE=256
NUM_EPOCHS=30
INIT_LR=0.0001

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

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = FDMobileNet()
    model.compile(optimizer=Adam(learning_rate=INIT_LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights
)

model.save('model.h5')
