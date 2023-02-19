'''
Created on 14-Feb-2023

@author: EZIGO
'''
from tensorflow.nn import max_pool_with_argmax
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.nn import log_softmax
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

class LeNet_5:
    @staticmethod
    def build(width, height, channels, classes, reg=0.002):
        inputShape = (width, height, channels)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (channels, height, width)
            chanDim = 1
        '''Input layer'''           
        inputs = Input(shape=inputShape)
        # x0 = BatchNormalization(axis=chanDim)(inputs)
        x0 = ZeroPadding2D(padding=(2, 2))(inputs)
        '''layer 1''' 
        x1 = Conv2D(filters=6, kernel_size=(5,5), strides=(2,2), padding='valid', use_bias=False)(x0)
        x1 = Activation("relu",name='x1')(x1)
        x1,x1_loc = max_pool_with_argmax(x1,ksize=(2, 2),strides=(2,2),padding='VALID',name='pooled_x1')
        # x1 = MaxPool2D(pool_size=(2, 2),strides=(2,2),padding='valid',name='pooled_x1')(x1)
        '''layer 2'''
        x2 = Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid',use_bias=False)(x1)
        x2 = Activation("relu",name='x2')(x2)
        x2,x2_loc = max_pool_with_argmax(x2,ksize=(2, 2),strides=(2,2),padding='VALID',name='pooled_x2')
        # x2 = MaxPool2D(pool_size=(2, 2),strides=(2,2),padding='valid',name='pooled_x2')(x2)
        '''layer3'''
        x2_vector = Flatten()(x2)
        x3 = Dense(120, kernel_regularizer=l2(reg))(x2_vector)
        x3 = Activation("relu")(x3)
        '''layer 4'''
        x4 = Dense(84, kernel_regularizer=l2(reg))(x3)
        x4 = Activation("relu")(x4)
        '''Output Layer'''
        x5 = Dense(classes, kernel_regularizer=l2(reg))(x4)
        x5 = Activation("softmax")(x5)
        
        model = Model(inputs=inputs, outputs=x5, name="LeNet_5")
        return model
    
# model = LeNet_5.build(28, 28, 1, 10, reg=0.002)
#
# model.summary()    
# plot_model(model,to_file="LeNet_5.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)      