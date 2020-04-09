"""
Changes:
 - Increased learning rate to 0.0005
 - 35 epochs
 - Increased dropout to 0.7
"""

from keras_squeezenet import SqueezeNet

from keras.layers import Flatten, GlobalAveragePooling2D, Dropout, Convolution2D, Dense, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

import tensorflow as tf

from collections import Counter

WIDTH, HEIGHT = (300, 300)

DROPOUT=0.7
CLASSES=10
BATCH_SIZE=32
NUM_EPOCHS=35
INIT_LR=0.0005

PATH = 'animals10/raw-img/'

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

base_model = SqueezeNet(input_shape=(HEIGHT, WIDTH, 3), weights=None, bypass='complex')
x = base_model.output

#x = Dropout(DROPOUT, name='drop9')(x)
#x = Convolution2D(CLASSES, (1, 1), padding='valid', name='conv10')(x)
#x = Activation('relu', name='relu_conv10')(x)
#x = GlobalAveragePooling2D()(x)
#predictions = Activation('softmax', name='loss')(x)

x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT, name='drop9')(x)
predictions = Dense(CLASSES, activation='softmax')(x)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer=Adam(lr=INIT_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    
model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights,
)

#results = model.evaluate_generator(generator=test_generator,
#                         steps=test_generator.samples // BATCH_SIZE)
#print('RESULTS:', {key: val for key, val in zip(model.metrics_names, results)})
    
model.save('model.h5')
