import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import  CategoricalCrossentropy
from tensorflow.keras.models import Sequential

from collections import Counter

PATH = '../Data/split_eccv/'
TRAIN_PATH = PATH + 'train'
TEST_PATH = PATH + 'test'

WIDTH, HEIGHT = (300, 300)
BATCH_SIZE=32
INIT_LR = 0.0005
NUM_EPOCHS=15
CLASSES=10
DROPOUT=0.4

VALIDATION_SPLIT=0.9

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
                             weights='imagenet')
    #base_model.trainable = False

    pool = GlobalAveragePooling2D()
    dropout = Dropout(DROPOUT)
    predictions = Dense(CLASSES, activation='softmax')
    model = Sequential([base_model, pool, dropout, predictions])
    model.compile(optimizer=RMSprop(lr=INIT_LR),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
                  

model.fit(
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
