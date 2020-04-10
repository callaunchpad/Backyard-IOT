WIDTH, HEIGHT = (300, 300)

DROPOUT=0.5
CLASSES=10
INIT_LR=0.001

BATCH_SIZE=16
NUM_EPOCHS=35

PATH = '../Data/animals10/raw-img/'

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, Activation, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, concatenate, add
from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from collections import Counter

import keras.backend as K
K.set_floatx('float16')
K.set_epsilon(1e-4) #default is 1e-7

def fire_module(x, fire_id, squeeze=16, expand=64):
    fire_id = 'fire' + str(fire_id) + '/'
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=fire_id + 'squeeze1x1')(x)
    x = Activation('relu', name=fire_id + 'relu_squeeze1x1')(x)
    
    left = Convolution2D(expand, (1, 1), padding='valid', name=fire_id + 'expand1x1')(x)
    left = Activation('relu', name=fire_id + 'relu_expand1x1')(left)
 
    right = Convolution2D(expand, (3, 3), padding='same', name=fire_id + 'expand3x3')(x)
    right = Activation('relu', name=fire_id + 'relu_expand3x3')(right)
 
    x = concatenate([left, right], axis=3, name=fire_id + 'concat')
    
    return x

def simple_add(from_layer, to_layer):
    return add([from_layer, to_layer])

def channels(layer):
    return layer.shape[3]

def complex_add(complex_id, from_layer, to_layer):
    complex_id = 'complex' + str(complex_id)
    from_layer = Convolution2D(channels(to_layer), (1, 1), padding='same', name=complex_id)(from_layer)
    from_layer = Activation('relu', name='relu_' + complex_id)(from_layer)
    return simple_add(from_layer, to_layer)


"""
Note: the simple and complex bypass connections feed into each other unless separated by a pooling layer.
"""
def SqueezeNet(input_shape, weights=None, bypass=None):
    
    if weights not in {None, 'base_model'}:
        raise ValueError('Unknown `weights` argument')
    
    if bypass == 'simple':
        simple = True
        complex = False
    elif bypass == 'complex':
        simple = complex = True
    elif bypass == None:
        simple = complex = False
    else:
        raise ValueError('Unknown `bypass` argument')
    
    inputs = Input(shape=input_shape, name='input')
    
    x = Convolution2D(64, 3, strides=2, padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = c = MaxPooling2D(pool_size=3, strides=2, name='maxpool1')(x)

    x = s = fire_module(x, fire_id=2, squeeze=16, expand=64)
    if complex:
        x = complex_add(1, c, x)
    x = c = fire_module(x, fire_id=3, squeeze=16, expand=64)
    if simple:
        x = simple_add(s, x)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    if complex:
        x = complex_add(2, c, x)
    x = s = MaxPooling2D(pool_size=3, strides=2, name='maxpool4')(x)
    
    x = c = fire_module(x, fire_id=5, squeeze=32, expand=128)
    if simple:
        x = simple_add(s, x)
    x = s = fire_module(x, fire_id=6, squeeze=48, expand=192)
    if complex:
        x = complex_add(3, c, x)
    x = c = fire_module(x, fire_id=7, squeeze=48, expand=192)
    if simple:
        x = simple_add(s, x)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    if complex:
        x = complex_add(4, c, x)
    x = s = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool8')(x)
    
    x = fire_module(s, fire_id=9, squeeze=64, expand=256)
    if simple:
        x = simple_add(s, x)
        
    #x = Dropout(DROPOUT, name='drop9')(x)
    
    x = Convolution2D(CLASSES, 1, strides=1, name='conv10')(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Activation('softmax', name='predictions')(x)
    
    model = Model(inputs, predictions, name='squeezenet')
    
    if weights == 'base_model':
        weights_path = 'base_model.h5'
    
    return model

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
    model = SqueezeNet(input_shape=(HEIGHT, WIDTH, 3), weights=None, bypass=bypass)
    model.compile(optimizer=RMSprop(lr=INIT_LR),
                  loss=CategoricalCrossentropy(from_logits=True),
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
