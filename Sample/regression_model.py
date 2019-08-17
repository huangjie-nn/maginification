#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# X=pd.read_pickle("/kaggle/input/datas.pickle")
# y=pd.read_pickle("/kaggle/input/labels.pickle")

X=pickle.load(open("datas.pickle","rb"))
y=pickle.load(open("labels.pickle","rb"))


y_float=np.array(y, dtype=np.float32)
X=X/255.0





# define base model
def baseline_model():
    model = Sequential()
    model.add(Flatten(input_shape=(224,224,1)))
    model.add(Dense(32, input_dim=50176, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)


kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y_float, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

