import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential

from collections import Counter

DIR='../Data/eccv'
WIDTH, HEIGHT, CHANNELS = (300, 300, 3)
VALIDATION_SPLIT=0.1

BATCH_SIZE=32
NUM_EPOCHS=10
INIT_LR = 0.001
CLASSES=10

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

base_model = ResNet50(weights=None, include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT)(x)
predictions = Dense(len(class_weights), activation= 'softmax')(x)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = MobileNetV2(input_shape=(HEIGHT, WIDTH, 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False

    pool = GlobalAveragePooling2D()
    predictions = Dense(CLASSES, activation='softmax')
    model = Sequential([base_model, pool, predictions])
    model.compile(optimizer=RMSprop(lr=INIT_LR),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights
)

model.save('model.h5')
