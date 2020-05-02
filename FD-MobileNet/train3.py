import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras_fdmobilenet import FDMobileNet
from collections import Counter
import os
import pickle
from prettytable import PrettyTable

BATCH_SIZE=32
NUM_EPOCHS=12
INIT_LR=1e-4
STEP=2
RATE=0.9

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

#Multiplies learning rate by RATE every STEP epochs, starting at INIT_LR
scheduler = lambda epoch: INIT_LR * (RATE ** (epoch // STEP))
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

strategy = tf.distribute.MirroredStrategy()

def train_and_save(alpha):
    with strategy.scope():
        model = FDMobileNet(alpha=alpha)
        model.compile(optimizer=Adam(learning_rate=0.0),
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

    RESULTS='results'
    if not os.path.exists(RESULTS):
        os.mkdir(RESULTS)

    model.save(os.path.join(RESULTS, 'fdmobilenet_{}x.h5'.format(alpha)))
    with open(os.path.join(RESULTS, 'history_{}x.pkl'.format(alpha)), 'wb') as f:
        pickle.dump(history.history, f)
        
    headers=['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss', 'lr']
    table = PrettyTable(headers)
    for i in range(NUM_EPOCHS):
        table.add_row([i+1] + [history.history[header][i] for header in headers[1:]])
    with open(os.path.join(RESULTS, 'table_{}x.txt'.format(alpha)), 'wb') as f:
        pickle.dump(str(table), f)

train_and_save(1)
train_and_save(0.5)
train_and_save(0.25)
