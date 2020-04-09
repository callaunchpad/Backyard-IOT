import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential

PATH = '../Data/animals10/raw-img/'

WIDTH, HEIGHT = (300, 300)
BATCH_SIZE=32
INIT_LR = 0.001
NUM_EPOCHS=10
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
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
                  
model.summary()

model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
)

model.save('model.h5')
