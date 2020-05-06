from common import *

with strategy.scope():
    model = FDMobileNet(input_shape=(HEIGHT, WIDTH, 3), classes=CLASSES, alpha=alpha)
    model.compile(optimizer=Adam(learning_rate=0.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

save()
