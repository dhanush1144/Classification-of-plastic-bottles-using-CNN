import tensorflow as tf
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

data_path='training'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))
img_size = 256
data = []
label = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            label.append(label_dict[category])

        except Exception as e:
            print('Exception:', e)

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
label=np.array(label)
from keras.utils import np_utils
new_label=np_utils.to_categorical(label)

x_train,x_test,y_train,y_test = train_test_split(data,new_label,test_size=0.2)

#show data

#training



#prediction
X = 10
img_size = 256
img_single = x_test[X]
img_single = cv2.resize(img_single, (img_size, img_size))
img_single = (np.expand_dims(img_single, 0))
img_single = img_single.reshape(img_single.shape[0],256,256,1)

model = models.load_model('C:/Users/2021PECAI240/Downloads/bottle.model')
print(model.predict(img_single))
print('prediction is',categories[np.argmax(y_test[X])])
plt.imshow(np.squeeze(img_single))
plt.xlabel(categories[np.argmax(y_test[X])] )

plt.grid(False)
plt.show()
