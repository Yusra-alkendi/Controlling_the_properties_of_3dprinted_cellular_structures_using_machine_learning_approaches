#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf 
import matplotlib 
from matplotlib import pyplot as plt 
import keras 
from keras import losses 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation 
from keras.initializers import orthogonal, normal
import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
import pandas as pd
    import numpy as np

   


Velocity =pd.read_csv('P_Velocity.csv')
Force = pd.read_csv('P_Force.csv')

V=pd.read_csv(Velocity)
F=pd.read_csv(Force)

X=np.array(V)
Y=np.array(F)   


scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples= scaler.fit_transform((X).reshape(-1,1))


scaler=MinMaxScaler(feature_range=[0,1])
scaled_train_samples1= scaler.fit_transform((Y).reshape(-1,1))

X_train,X_test,y_train,y_test=train_test_split(scaled_train_samples,scaled_train_samples1, test_size=0.15)


# In[8]:


#create model
model = Sequential()
model.add(Dense(18,input_dim=1,kernel_initializer='orthogonal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(18,kernel_initializer='orthogonal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(18,kernel_initializer='orthogonal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(18,kernel_initializer='orthogonal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(18,kernel_initializer='orthogonal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(18,kernel_initializer='orthogonal', activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='tanh'))


# In[45]:


#compile 
adam=keras.optimizers.Adam(lr=0.000001,beta_1=0.9, beta_2=0.999)
model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mse'])

#fit
history=model.fit(X_train, y_train,validation_split=0.15, epochs=1000)


# In[46]:


#evaluate
mse_value=model.evaluate(X_test,y_test)
print(mse_value)


# In[52]:


print (model.summary())


# In[47]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[59]:


# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


y_pred=model.predict(X_test)
from sklearn.metrics import r2_score 
r2_score(y_test,y_pred)


# In[40]:


y_pred=model.predict(scaled_train_samples)
y_pred_inverse = scaler.inverse_transform(y_pred)
print(y_pred_inverse)

