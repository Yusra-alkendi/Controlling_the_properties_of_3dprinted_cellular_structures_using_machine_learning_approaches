#!/usr/bin/env python
# coding: utf-8

# In[64]:


import keras
from keras import losses
from keras import metrics
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.initializers import Orthogonal
from keras.metrics import categorical_crossentropy 
import matplotlib 
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from numpy import array
                    


# In[102]:


P_Velocity = (r'C:\Users\Acer\Desktop\Yousef\FINAL PROJECT\data\pid velocity.csv')
P_Force=(r'C:\Users\Acer\Desktop\Yousef\FINAL PROJECT\data\pid force.csv')
Velocity=pd.read_csv(P_Velocity)
Force=pd.read_csv(P_Force)
V=np.array(Velocity[:], dtype=np.float)
F=np.array(Force[10:], dtype=np.float)
samples=list()
length=200
for i in range(0,1000,length):
    sample=V[i:i+length]
    samples.append(sample)


# In[121]:


def split_sequence(sequence, n_steps):
  X=list()
  for i in range(len(sequence)):
        end_ix= i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x = sequence[i:end_ix]
        X.append(seq_x)
  return array(X)
X=split_sequence(V,10)
print(X)


# In[104]:


print(F.shape)


# In[113]:


model=Sequential()
model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[114]:


model.fit(X,F,epochs=200)


# In[115]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[120]:


model.summary()

