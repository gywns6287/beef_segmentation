from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.models import *
import os 

################Unet structure###########################
def Unet_identity(layer,channel,index):
    layer = Conv2D(channel , (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = 'unet'+index.pop(0))(layer)
    layer = Conv2D(channel , (3,3), activation = 'relu', padding = 'same', 
                   kernel_initializer = 'he_normal', name = 'unet'+index.pop(0))(layer)
    return layer


def Unet_concat_identity(layers,channel,index):
    merge = concatenate(layers,axis=3)
    layer = Unet_identity(merge,channel,index)
    return layer 

def Unet(input_shape = (224,224,3)):
    
    index = list([str(i) for i in range(1,101)])

    input_tensor = Input(shape=input_shape, dtype='float32')

    conv1 = Unet_identity(input_tensor, 64, index)
    down1 = MaxPool2D(pool_size = (2,2))(conv1)
    
    conv2 = Unet_identity(down1, 128, index)
    down2 = MaxPool2D(pool_size = (2,2))(conv2)
    
    conv3 = Unet_identity(down2, 256, index)
    down3 = MaxPool2D(pool_size = (2,2))(conv3)

    conv4 = Unet_identity(down3, 512, index)
    down4 = MaxPool2D(pool_size = (2,2))(conv4)
    
    conv5 = Unet_identity(down4, 1024, index)
    up5 = UpSampling2D(size = (2,2))(conv5)

    conv4 = Unet_concat_identity([conv4,up5],512, index)
    up4 = UpSampling2D(size = (2,2))(conv4)

    conv3 = Unet_concat_identity([conv3,up4],256, index)
    up3 = UpSampling2D(size = (2,2))(conv3)

    conv2 = Unet_concat_identity([conv2,up3],128, index)
    up2 = UpSampling2D(size = (2,2))(conv2)

    conv1 = Unet_concat_identity([conv1,up2],64, index)
    unet_out = Conv2D(1, (1,1), activation = 'sigmoid',name = 'unet_out')(conv1)

    return Model(inputs = input_tensor, outputs = unet_out)
#############################################################
