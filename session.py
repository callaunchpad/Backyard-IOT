import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.optimizers import Adam
from keras.models import Model


from collections import Counter

DIR='data'
WIDTH, HEIGHT, CHANNELS = (512, 384, 3)
VALIDATION_SPLIT=4800/57864

DROPOUT = 0.3

BATCH_SIZE=24
NUM_EPOCHS=500
INIT_LR = 0.001

datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT)

train_generator = datagen.flow_from_directory(
    DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    subset='training')

validation_generator = datagen.flow_from_directory(
    DIR,
    target_size=(HEIGHT, WIDTH),
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
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer=Adam(lr=INIT_LR), loss='categorical_crossentropy', metrics=['accuracy'])
    
model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights,
    shuffle=True
)

model.evaluate_generator(generator=validation_generator,
steps=validation_generator.samples // BATCH_SIZE)
    
model.save('model.h5')
