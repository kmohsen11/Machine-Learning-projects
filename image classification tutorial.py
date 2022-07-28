#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[55]:


inputs = np.loadtxt(r"C:\Users\Nayeb\Downloads\Image Classification CNN Keras Dataset\input.csv", delimiter = ',')
inputs_test = np.loadtxt(r"C:\Users\Nayeb\Downloads\Image Classification CNN Keras Dataset\input_test.csv", delimiter = ',')
labels = np.loadtxt(r"C:\Users\Nayeb\Downloads\Image Classification CNN Keras Dataset\labels.csv", delimiter = ',')
labels_test = np.loadtxt(r"C:\Users\Nayeb\Downloads\Image Classification CNN Keras Dataset\labels_test.csv", delimiter = ',')


# In[56]:


inputs = inputs.reshape(len(inputs), 100, 100, 3)
inputs_test = inputs_test.reshape(len(inputs_test), 100, 100, 3)
labels = labels.reshape(len(labels), 1)
labels_test = labels_test.reshape(len(labels_test), 1)
inputs = inputs / 255.0
inputs_test = inputs_test/ 255.0
labels = labels / 255.0
labels_test = labels_test / 255.0


# In[57]:


print(inputs.shape)
print(inputs_test.shape)
print(labels.shape)
print(labels_test.shape)


# In[58]:


plt.imshow(inputs[1,:])


# In[59]:


model= keras.Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (100,100,3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64, activation= 'relu'),
    Dense(1, activation = "sigmoid")
    
])


# In[60]:


opt = keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


# In[61]:


model.fit(inputs, labels, epochs = 10, batch_size = 64)


# In[62]:


model.evaluate(inputs, labels)


# In[63]:


idx2 = random.randint(0, len(labels))
plt.imshow(inputs[idx2, :])
plt.show()

y_predict = model.predict(inputs[idx2, :].reshape(1, 100, 100, 3))
y_predict = y_predict >0.5
if(y_predict == 0):
    y_predict = 'dog'
else:
    y_predict = "cat"
    
print (y_predict)


# In[ ]:





# In[ ]:




