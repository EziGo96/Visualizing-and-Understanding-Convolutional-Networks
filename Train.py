'''
Created on 15-Feb-2023

@author: EZIGO
'''
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from LeNet_5 import LeNet_5
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import save_model
from Load_Mnist import load_mnist
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler



'''training and validation data'''
X_train, y_train = load_mnist('DATA/MNIST', kind='train')

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

print(X_train.shape)
print(X_valid.shape)
print(y_train)
print(y_valid)

# plt.figure()
# image = X_train[0]
# plt.imshow(image, cmap='gray')
# plt.figure()
# image = X_valid[0]
# plt.imshow(image, cmap='gray')




'''EDA'''
# print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
# print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
#
# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.savefig('mnist_all.png', dpi=300)
#
# fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()
# for i in range(25):
#     img = X_train[y_train == 7][i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.savefig('mnist_7.png', dpi=300)
# Turn our scalar targets into binary categories

'''pre-processing'''
classes = 10
y_train = to_categorical(y_train, classes)
y_valid = to_categorical(y_valid, classes)

print(y_train.shape)
print(y_valid.shape)
# Normalize our image data
X_train = X_train / 255
X_valid = X_valid / 255

'''training'''
# datagen = ImageDataGenerator(
#     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#     zoom_range=0.1,  # Randomly zoom image
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=False,  # Don't randomly flip images horizontally
#     vertical_flip=False, # Don't randomly flip images vertically
# )


BS= 32
NUM_EPOCHS =20

# def polynomial_decay(epoch):
#     maxEpochs = NUM_EPOCHS
#     baseLR = INIT_LR
#     power = 1
#     alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
#     return alpha

mcp_save = ModelCheckpoint("LeNet_5_wts_.hdf5", save_best_only=True, monitor='val_loss', mode='min')
callbacks = [mcp_save]
# img_iter = datagen.flow(X_train, y_train, batch_size=BS)
# datagen.fit(X_train)

model = LeNet_5.build(28, 28, 1 ,classes, reg=0.002)

model.summary()    
plot_model(model,to_file="LeNet_5.png",show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True)
opt = Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])  

H = model.fit(X_train,
              y_train,
              epochs=NUM_EPOCHS,
              steps_per_epoch=len(X_train)/BS, # Run same number of steps we would if we were not using a generator.
              validation_data=(X_valid, y_valid),
              callbacks=callbacks)

df=pd.DataFrame()
df.from_dict(H.history).to_csv("Training.csv",index=False)
model_json = model.to_json(indent=3)
with open("Lenet_5_architecture.json", "w") as json_file:
    json_file.write(model_json)
save_model(model, "LeNet_5.hp5", save_format="h5")



# x, y = img_iter.next()
# fig, ax = plt.subplots(nrows=4, ncols=8)
# for i in range(BS):
#     image = x[i]
#     ax.flatten()[i].imshow(np.squeeze(image),cmap='gray')

plt.show()

