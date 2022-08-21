

import os
import pathlib
# Dataframes and matrices ---------------------------------------
import numpy as np
import pandas as pd
# Graphics ------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Machine learning ----------------------------------------------
from sklearn.model_selection import train_test_split
# Deep learning -------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import datasets, layers, models
from keras.utils import load_img
from keras.utils import img_to_array
from keras.utils import get_file
from keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.models import load_model

def load_img(filename):
    img = cv2.imread(filename)
    try:
        img_resized = cv2.resize(img,(200,200))
    except:
        print(f'Error in image {img}')
    return img_resized


filename = './cat-dog_Model.h5'
img = tf.keras.utils.load_img("10.jpg",target_size=(200,200))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
saved_model= load_model(filename)
output =saved_model.predict(img)

if output[0][0] > output[0][1]:
    print("cat")
else:
    print('dog')
    
