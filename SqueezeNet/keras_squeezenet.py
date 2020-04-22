from keras.models import Model
from keras.layers import Input, Convolution2D, Dropout, Activation, MaxPooling2D, GlobalAveragePooling2D, concatenate, add

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

def complex_add(from_layer, to_layer):
    from_layer = Convolution2D(channels(to_layer), (1, 1), padding='same')(from_layer)
    from_layer = Activation('relu')(from_layer)
    return simple_add(from_layer, to_layer)

def SqueezeNet(input_shape, classes=1000, bypass=None):
       
    if bypass == 'complex':
        complex = simple = True
    elif bypass == 'simple':
        simple = True
        complex = False
    elif bypass == None:
        complex = simple = False
    else:
        raise ValueError('Unknown `bypass` argument')
    
    inputs = Input(shape=input_shape)
    
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    c = x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    s = x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    if complex:
        x = complex_add(c, x) #complex conv1 -> fire3
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    if simple:
        x = simple_add(s, x) #simple fire2 -> fire4
    c = x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
    
    s = x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    if complex:
        x = complex_add(c, x) #complex fire3 -> fire5
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    if simple:
        x = simple_add(s, x) #simple fire4 -> fire6
    c = x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    
    s = x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    if complex:
        x = complex_add(c, x) #complex fire5 -> fire7
    c = x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    if simple:
        x = simple_add(s, x) #simple fire6 -> fire8
    s = x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    if complex:
        x = complex_add(c, x) #complex fire7 -> fire9
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    if simple:
        x = simple_add(s, x) #simple fire8 -> conv10
    
    x = Dropout(0.5, name='drop9')(x)
    x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)

    model = Model(inputs, x, name='squeezenet')
    
    return model