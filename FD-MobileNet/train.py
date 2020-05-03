import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_fdmobilenet import FDMobileNet
from collections import Counter
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

BATCH_SIZE=256
NUM_EPOCHS=50
INIT_LR=1e-4
STEP=10
RATE=0.8

IMAGES = '../Data/iwildcam/images'
LABELS = '../Data/iwildcam/labels.csv'
WIDTH, HEIGHT = (256, 187)


RESULTS='results'
if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)

df = pd.read_csv(LABELS)
CLASSES=list(df.label.unique())

df, discard = train_test_split(df, train_size=0.2)
train_df, test_df = train_test_split(df, test_size=0.1)
train_df, val_df = train_test_split(train_df, test_size=0.1)
train_df.to_csv(os.path.join(RESULTS, 'train.csv'), index=False)
val_df.to_csv(os.path.join(RESULTS, 'val.csv'), index=False)
test_df.to_csv(os.path.join(RESULTS, 'test.csv'), index=False)

del df, discard


train_datagen = ImageDataGenerator(
    rescale=1/.255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)
test_datagen = ImageDataGenerator(
    rescale=1/.255)

train_generator = train_datagen.flow_from_dataframe(
    classes=CLASSES,
    dataframe=train_df,
    directory=IMAGES,
    target_size=(HEIGHT, WIDTH),
    x_col='file_name',
    y_col='label',
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
)

test_generator = test_datagen.flow_from_dataframe(
    classes=CLASSES,
    dataframe=test_df,
    directory=IMAGES,
    target_size=(HEIGHT, WIDTH),
    x_col='file_name',
    y_col='label',
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
)

val_generator = test_datagen.flow_from_dataframe(
    classes=CLASSES,
    dataframe=val_df,
    directory=IMAGES,
    target_size=(HEIGHT, WIDTH),
    x_col='file_name',
    y_col='label',
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
)


#Calculate class weights
counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

callbacks = []

#Multiplies learning rate by RATE every STEP epochs, starting at INIT_LR
scheduler = lambda epoch: INIT_LR * (RATE ** (epoch // STEP))
callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))

#checkpoint
callbacks.append(ModelCheckpoint(os.path.join(RESULTS, "checkpoint.h5"), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = FDMobileNet(input_shape=(HEIGHT, WIDTH, 3), classes=len(CLASSES), alpha=1)
    model.compile(optimizer=Adam(learning_rate=0.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights, callbacks=callbacks
)

model.save(os.path.join(RESULTS, 'fdmobilenet_{}x.h5'.format(alpha)))

evaluation = model.evaluate(
    test_generator,
    steps = test_generator.samples // BATCH_SIZE
)

with open(os.path.join(RESULTS, 'history_{}x.pkl'.format(alpha)), 'wb') as f:
    pickle.dump(history.history, f)
    
headers=['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss', 'lr']
table = PrettyTable(headers)
for i in range(NUM_EPOCHS):
    table.add_row([i+1] + [history.history[header][i] for header in headers[1:]])
with open(os.path.join(RESULTS, 'table_{}x.txt'.format(alpha)), 'w') as f:
    f.write(str(table))
    f.write('\n\nTest Results:')
    f.write('loss:', evaluation[0])
    f.write('accuracy:', evaluation[1])
