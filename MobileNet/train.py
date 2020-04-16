import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.losses import  CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from collections import Counter

PATH = '../Data/split_eccv/'
TRAIN_PATH = PATH + 'train'
TEST_PATH = PATH + 'test'

WIDTH, HEIGHT = (500, 500)
BATCH_SIZE=16
INIT_LR = 0.0004
NUM_EPOCHS=50
CLASSES=16
DROPOUT=0.5

VALIDATION_SPLIT=0.1

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='validation')

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(HEIGHT, WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='training')


counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = MobileNetV2(input_shape=(HEIGHT, WIDTH, 3),
        include_top=False,
        weights=None)
    #base_model = load_model('base_model.h5')
    #base_model.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)
    predictions = Dense(CLASSES, activation='softmax')(x)
    model = model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights
)

results = model.evaluate_generator(generator=test_generator, 
    steps=test_generator.samples // BATCH_SIZE)

print('RESULTS:', {key:val for key, val in zip(model.metrics_names, results)})

model.save('model.h5')
