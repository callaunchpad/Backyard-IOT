from tensorflow.keras.layers import Input, Dense, ReLU, Conv2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def FDMobileNet(input_shape=(224, 224, 3), classes=10, alpha=1):
    
    
    """Helpful custom keras blocks"""
    
    def ConvBlock(layer, filters, kernel_size=3, strides=1, **kwargs):
        def block(x):
            x = layer(filters, kernel_size, strides, padding='same', use_bias=False, **kwargs)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x
        return block

    SeparableBlock = lambda filters, strides=1: ConvBlock(SeparableConv2D, filters, strides=strides, depth_multiplier=alpha)
    
    
    """Build the model"""
    
    inputs = Input(shape=input_shape)
    
    x = ConvBlock(Conv2D, filters=32, strides=2)(inputs)
    
    x = SeparableBlock(filters=64, strides=2)(x)
    
    x = SeparableBlock(filters=128, strides=2)(x)
    x = SeparableBlock(filters=128)(x)
    
    x = SeparableBlock(filters=256, strides=2)(x)
    x = SeparableBlock(filters=256)(x)
    
    x = SeparableBlock(filters=512, strides=2)(x)
    for i in range(4):
        x = SeparableBlock(filters=512)(x)
    x = SeparableBlock(filters=1024)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs, x, name='FD-MobileNet')
    

    return model
