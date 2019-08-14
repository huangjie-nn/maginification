#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D






X=pickle.load(open("datas.pickle","rb"))
y=pickle.load(open("labels.pickle","rb"))

y_coded=y.codes

print(y_coded)
X=X/255.0


model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1000))
model.add(Activation("relu"))

model.compile(loss="categorical_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

model.fit(X,y_binary,batch_size=32,validation_split=0.2)


# In[ ]:





