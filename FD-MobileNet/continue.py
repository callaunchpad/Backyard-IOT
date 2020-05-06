MODEL_PATH = 'checkpoint.h5'

with strategy.scope():
    model = FDMobileNet(input_shape=(HEIGHT, WIDTH, 3), classes=len(CLASSES), alpha=alpha)
    model.load_weights(MODEL_PATH)
    model.compile(optimizer=Adam(learning_rate=0.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

save()
