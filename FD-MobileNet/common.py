import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--alpha')
parser.add_argument('--gpus')
args = vars(parser.parse_args())
alpha = int(args['alpha'])
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_fdmobilenet import FDMobileNet
from collections import Counter
import pickle
from prettytable import PrettyTable

BATCH_SIZE=32
NUM_EPOCHS=30
INIT_LR=1e-4
STEP=10
RATE=0.8

WIDTH, HEIGHT = (224, 224)
CLASSES=10
PATH = '/home/memerling/Backyard-IOT/Data/animals10/raw-img/'

datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.05,
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)

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

callbacks = []

#Multiplies learning rate by RATE every STEP epochs, starting at INIT_LR
scheduler = lambda epoch: INIT_LR * (RATE ** (epoch // STEP))
callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

#checkpoint
callbacks.append(ModelCheckpoint("checkpoint.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

strategy = tf.distribute.MirroredStrategy()
