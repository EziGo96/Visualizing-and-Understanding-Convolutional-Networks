'''
Created on 17-Feb-2023

@author: EZIGO
'''
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import numpy as np
from sklearn import metrics
from Load_Mnist import load_mnist
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2DTranspose
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

'''test data'''
X_test, y_test = load_mnist('DATA/MNIST', kind='t10k')
y_test=y_test.tolist()
#indices of 0,1,5,8
print(y_test.index(0),y_test.index(1),y_test.index(5),y_test.index(8))
BS=32
X_test = X_test / 255
'''model testing'''
print("LeNet_5 Classifier")
json_file = open("Lenet_5_architecture.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("LeNet_5_wts_.hdf5")
y_preds_proba = model.predict(X_test,batch_size = BS)
y_preds=np.argmax(y_preds_proba,axis=1)
y_pred_proba = y_preds_proba[::,1]
print(metrics.classification_report(y_test, y_preds))
print(metrics.confusion_matrix(y_test, y_preds))

'''visualisation'''
#test input
plt.figure()
test_images = [X_test[3],X_test[2],X_test[8],X_test[61]]
for test_image in test_images:
    y_preds_proba = model.predict(np.array([test_image]),batch_size = 1)
    y_preds=np.argmax(y_preds_proba,axis=1)
    print(y_preds)
    plt.imshow(test_image, cmap='gray')
    
    '''features from convolution layers'''
    feature_extractor_layer1 = keras.Model(inputs=model.inputs, outputs=model.get_layer(name="x1").output)
    feature_extractor_layer1_pooled = keras.Model(inputs=model.inputs, outputs=model.get_layer(name="tf.nn.max_pool_with_argmax").output)
    feature_extractor_layer2 = keras.Model(inputs=model.inputs, outputs=model.get_layer(name="x2").output)
    feature_extractor_layer2_pooled = keras.Model(inputs=model.inputs, outputs=model.get_layer(name="tf.nn.max_pool_with_argmax_1").output)
    
    #conv1 activated unpooled feature
    x1=feature_extractor_layer1(np.array([test_image]))
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(6):
        ax[i].imshow(x1[0][:,:,i],cmap='gray')
    
    #conv1 activated pooled feature
    pooled_x1,pooled_x1_mask=feature_extractor_layer1_pooled(np.array([test_image]))
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(6):
        ax[i].imshow(pooled_x1[0][:,:,i],cmap='gray')
    
    #conv2 activated unpooled feature
    x2=feature_extractor_layer2(np.array([test_image]))
    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(16):
        ax[i].imshow(x2[0][:,:,i],cmap='gray')
    
    #conv2 activated pooled feature
    pooled_x2,pooled_x2_mask=feature_extractor_layer2_pooled(np.array([test_image]))
    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(16):
        ax[i].imshow(pooled_x2[0][:,:,i],cmap='gray',vmin=0,vmax=31)
        
    #conv1 kernel
    def conv1_w1(shape, dtype=None):
        return K.variable(model.get_layer(name="conv2d").get_weights()[0])
    #conv2 kernel
    def conv2_w2(shape, dtype=None):
        return K.variable(model.get_layer(name="conv2d_1").get_weights()[0])  
    
    '''deconvnet for layer1 activations'''
    x0_deconv1=tfa.layers.MaxUnpooling2DV2((1, 14, 14, 6))(pooled_x1,pooled_x1_mask)
    x0_deconv1=keras.layers.Activation('relu')(x0_deconv1) 
    x0_deconv1=Conv2DTranspose(filters=1,kernel_size=(5,5), strides=(2,2), padding='valid', use_bias=False, kernel_initializer=conv1_w1)(x0_deconv1)
    plt.figure()
    plt.imshow(x0_deconv1[0],cmap='gray')
    
    '''deconvnet for layer2 activations'''
    x1_deconv2=tfa.layers.MaxUnpooling2DV2((1, 3, 3, 16))(pooled_x2,pooled_x2_mask)
    x1_deconv2=keras.layers.Activation('relu')(x1_deconv2) 
    x1_deconv2=Conv2DTranspose(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid',use_bias=False, kernel_initializer=conv2_w2)(x1_deconv2)
    x0_deconv2=tfa.layers.MaxUnpooling2DV2((1, 14, 14, 6))(x1_deconv2,pooled_x1_mask)
    x0_deconv2=keras.layers.Activation('relu')(x0_deconv2) 
    x0_deconv2=Conv2DTranspose(filters=1,kernel_size=(5,5), strides=(2,2), padding='valid', use_bias=False, kernel_initializer=conv1_w1)(x0_deconv2)
    
    plt.figure()
    plt.imshow(x0_deconv2[0],cmap='gray')
    plt.show()    