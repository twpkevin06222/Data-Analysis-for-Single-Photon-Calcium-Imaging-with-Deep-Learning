import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax
from coord_conv import CoordConv

#references: https://github.com/titu1994/keras-coordconv/blob/master/experiments/train_uniform_classifier.py
"""
- This implementation is a replicate of coordconv uniform dataset classification 
task from the paper "An intriguing failing of convolutional neural networks and 
the CoordConv solution" by R.Liu et al. (2018) from Uber AI. 

- The model takes coordinates as an input
"""

# To make data, please run the code: coordconv_make_data.py
train_onehot = np.load('coordconv_data/train_onehot.npy').astype('float32')
test_onehot = np.load('coordconv_data/test_onehot.npy').astype('float32')

#retrieve coordinates
coord_train = np.where(train_onehot == 1.0)
coord_test = np.where(test_onehot == 1.0)

#Training coords
x_coord_train = coord_train[1]
y_coord_train = coord_train[2]

#Test coords
x_coord_test = coord_test[1]
y_coord_test = coord_test[2]

train_set = np.zeros((len(x_coord_train), 1, 1, 2), dtype='float32')
test_set = np.zeros((len(x_coord_test), 1, 1, 2), dtype='float32')

for i, (x, y) in enumerate(zip(x_coord_train, y_coord_train)):
    train_set[i, 0, 0, 0] = x  # 1st channel x coord
    train_set[i, 0, 0, 1] = y  # 2nd channel y coord

for i, (x, y) in enumerate(zip(x_coord_test, y_coord_test)):
    test_set[i, 0, 0, 0] = x  # 1st channel x coord
    test_set[i, 0, 0, 1] = y  # 2nd channel y coord

train_set = np.tile(train_set, [1, 64, 64, 1])
# train_set.shape[0] (64, 64, 2) => return: 1st channel, 64 numbers of same x_coord ..
test_set = np.tile(test_set, [1, 64, 64, 1])

#normalizing dataset

train_set /= (64. - 1.)
test_set /= (64. - 1.)
print('Train set : shape: {}, max value:{}, min value: {}'.format(train_set.shape, train_set.max(), train_set.min()))
print('Test set : : shape: {}, max value:{}, min value: {} '.format(test_set.shape, test_set.max(), test_set.min()))

# Plot dataset

plt.imshow(np.sum(train_onehot, axis=0)[:, :, 0], cmap='gray')
plt.title('Train One-hot dataset')
plt.show()
plt.imshow(np.sum(test_onehot, axis=0)[:, :, 0], cmap='gray')
plt.title('Test One-hot dataset')
plt.show()

#flatten one hot labels for final layer
train_onehot = train_onehot.reshape((-1, 64 * 64)) #shape(batch, 64*64)
test_onehot = test_onehot.reshape((-1, 64 * 64))

#model
model = Sequential([
    CoordConv(x_dim = 64, y_dim = 64, with_r = False, filters = 64,
              kernel_size = 1, padding='same', activation='relu'),
    Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = "same", activation = 'relu'),
    Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = "same", activation = 'relu'),
    Conv2D(filters = 64, kernel_size = 1, strides = 1, padding = "same", activation = 'relu'),
    Conv2D(filters = 64, kernel_size = 1, strides = 1, padding = "same", activation = 'relu'),
    Conv2D(filters = 1, kernel_size = 1, strides = 1, padding = "same" ),
    Flatten(),
    Softmax(axis = -1),
])

model.build((None, 64,64, 2))

#model input coordinates

optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
model.fit(train_set, train_onehot, batch_size = 32, epochs = 10,
          verbose = 1, validation_data = (test_set, test_onehot))

#Visualize test set
preds = model.predict(test_set)
print(np.min(preds), np.max(preds))

preds = preds.reshape((-1, 64, 64, 1))

plt.imshow(np.sum(preds, axis=0)[:, :, 0], cmap='gray')
plt.title('Predictions')
plt.show()

scores = model.evaluate(test_set, test_onehot, batch_size=128, verbose=1)

print()
for name, score in zip(model.metrics_names, scores):
    print(name, score)