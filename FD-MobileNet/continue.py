MODEL_PATH = 'checkpoint.h5'

with strategy.scope():
    model = FDMobileNet(input_shape=(HEIGHT, WIDTH, 3), classes=CLASSES, alpha=alpha)
    model.load_weights(MODEL_PATH)
    model.compile(optimizer=Adam(learning_rate=0.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    class_weight=class_weights, callbacks=callbacks
)

model.save( 'model.h5')

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
    
headers=['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss', 'lr']
table = PrettyTable(headers)
for i in range(NUM_EPOCHS):
    table.add_row([i+1] + [history.history[header][i] for header in headers[1:]])
with open('table.txt', 'w') as f:
    f.write(str(table))
