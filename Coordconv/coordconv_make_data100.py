import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D
from sklearn.model_selection import train_test_split

#refence: https://github.com/titu1994/keras-coordconv/blob/master/experiments/make_dataset.py

#create folder to store data
if not os.path.exists('coordconv_data/'):
    os.makedirs('coordconv_data/')

# From https://arxiv.org/pdf/1807.03247.pdf code snippet
onehots = np.pad(np.eye(8464, dtype = 'float32').reshape((8464, 92, 92, 1)), ((0,0), (4,4), (4,4), (0,0)), "constant")

model = Sequential([
    Conv2D(filters = 1, kernel_size = 9, strides = 1, padding = "same" )
])

model.build((None, 100, 100, 1))

image = model(onehots)
image = np.asarray(image)

#uniform dataset
indices = np.arange(0, len(onehots), dtype='int32')
train, test = train_test_split(indices, test_size=0.2, random_state=0)

train_onehot = onehots[train]
train_images = image[train]

test_onehot = onehots[test]
test_images = image[test]

np.save('coordconv_data/train_onehot100.npy', train_onehot)
np.save('coordconv_data/train_images100.npy', train_images)
np.save('coordconv_data/test_onehot100.npy', test_onehot)
np.save('coordconv_data/test_images100.npy', test_images)
