import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv





DATADIR = '/home/hp/Desktop/tensor-task8/'
CATEGORIES = ['Human','Non-Human']

IMG_SIZE = 64


training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img),0)
                new_array = cv.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()


import random
random.shuffle(training_data)

X = []
y = []


for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y = np.array(y)

X = X/255.0


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dropout, Activation


model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:],activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:],activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y,batch_size=32,validation_split=0.2)










