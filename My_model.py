import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
import tensorflow.keras as keras
from tensorflow.keras.layers import Lambda




def ResizeLayerLike(tensorA,Size):
    sB =Size 
    def resize_like(tensor, sB): return tf.image.resize(tensor, sB[1:3])
    return Lambda(resize_like, arguments={'sB':sB})(tensorA)

def my_model(input_size = (224,224,3)):
    
    inputs = keras.layers.Input(input_size)
 
    x = keras.layers.SeparableConv2D(32, (3, 3),padding='same',strides=(1, 1))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x) 
    x = keras.layers.SeparableConv2D(32, (3, 3),padding='same',strides=(1, 1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x_=  keras.layers.Conv2D(32, 1, padding='same')(x)
    x_ = keras.layers.BatchNormalization()(x_)
    x_ = keras.layers.Activation('relu')(x_) 
    x_1 = keras.layers.Conv2D(32, (3, 3),strides=(2, 2), padding='same')(x_)  
    
    Resized_img_1=ResizeLayerLike(inputs,(0,112,112,3))
    conc1 = keras.layers.concatenate([x_1, Resized_img_1],axis=-1)
    
    
    x = keras.layers.SeparableConv2D(64, (3, 3),padding='same',strides=(1, 1))(conc1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x) 
    x = keras.layers.SeparableConv2D(64, (3, 3),padding='same',strides=(1, 1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x_=  keras.layers.Conv2D(32, 1, padding='same')(x)
    x_ = keras.layers.BatchNormalization()(x_)
    x_ = keras.layers.Activation('relu')(x_) 
    x_2 = keras.layers.Conv2D(32, (3, 3),strides=(2, 2), padding='same')(x_)  
    
    Resized_img_2=ResizeLayerLike(inputs,(0,56,56,3))
    conc2 = keras.layers.concatenate([x_2, Resized_img_2],axis=-1)
    
    
    x = keras.layers.SeparableConv2D(128, (3, 3),padding='same',strides=(1, 1))(conc2)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x) 
    x = keras.layers.SeparableConv2D(128, (3, 3),padding='same',strides=(1, 1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x_=  keras.layers.Conv2D(32, 1, padding='same')(x)
    x_ = keras.layers.BatchNormalization()(x_)
    x_ = keras.layers.Activation('relu')(x_) 
    x_3 = keras.layers.Conv2D(32, (3, 3),strides=(2, 2), padding='same')(x_)  
    
    Resized_img_3=ResizeLayerLike(inputs,(0,28,28,3))
    conc3 = keras.layers.concatenate([x_3, Resized_img_3],axis=-1)
    
    
    
    x = keras.layers.SeparableConv2D(256, (3, 3),padding='same',strides=(1, 1))(conc3)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x) 
    x = keras.layers.SeparableConv2D(256, (3, 3),padding='same',strides=(1, 1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x_=  keras.layers.Conv2D(32, 1, padding='same')(x)
    x_ = keras.layers.BatchNormalization()(x_)
    x_ = keras.layers.Activation('relu')(x_) 
    x_4 = keras.layers.Conv2D(32, (3, 3),strides=(2, 2), padding='same')(x_)  
    
    Resized_img_4=ResizeLayerLike(inputs,(0,14,14,3))
    conc4 = keras.layers.concatenate([x_4, Resized_img_4],axis=-1)


    x = keras.layers.SeparableConv2D(512, (3, 3),padding='same',strides=(1, 1))(conc4)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x) 
    x = keras.layers.SeparableConv2D(512, (3, 3),padding='same',strides=(1, 1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)sssss

    x_=  keras.layers.Conv2D(32, 1, padding='same')(x)
    x_ = keras.layers.BatchNormalization()(x_)
    x_ = keras.layers.Activation('relu')(x_) 
    x_5 = keras.layers.Conv2D(32, (3, 3),strides=(2, 2), padding='same')(x_)  
    
    Resized_img_5=ResizeLayerLike(inputs,(0,7,7,3))
    conc5 = keras.layers.concatenate([x_5, Resized_img_5],axis=-1)
    
    
    
    x_=  keras.layers.Conv2D(64, 3, padding='same')(conc5)
    x_ = keras.layers.BatchNormalization()(x_)
    x_ = keras.layers.Activation('relu')(x_) 
    
    x_=  keras.layers.Conv2D(64, 3, padding='same')(x_)
    x_ = keras.layers.BatchNormalization()(x_)
    x_ = keras.layers.Activation('relu')(x_) 
    
    
    x_=  keras.layers.Conv2D(32, 1, padding='same')(x_)
    
    
    
    model = keras.models.Model(inputs=inputs, outputs=conc5)
    return model






model=my_model()
model.summary()




