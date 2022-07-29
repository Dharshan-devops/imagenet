#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
import scipy
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


imagegen = ImageDataGenerator()


# In[4]:


train = imagegen.flow_from_directory("imagenette2/train/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))


# In[5]:


val = imagegen.flow_from_directory("imagenette2/val/", class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))


# In[6]:


model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))


# In[7]:


model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))


# In[8]:


model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())


# In[9]:


model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2,2), padding='valid'))
model.add(BatchNormalization())


# In[10]:


model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))


# In[11]:


model.add(Dense(units=10, activation='softmax'))


# In[12]:


model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# 

# In[13]:


model.summary()


# In[15]:


import scipy
epochs=30
history=model.fit(train,validation_data=val,epochs=epochs)


# In[17]:


from keras.applications import VGG16

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=False, weights='imagenet')
pretrained_model.summary()


# In[19]:


from keras.utils import to_categorical


# In[ ]:


vgg_features_train = pretrained_model.predict(train)
vgg_features_val = pretrained_model.predict(val)


# In[22]:


train_target = to_categorical(train.labels)
val_target = to_categorical(val.labels)


# In[24]:


model2 = Sequential()
model2.add(Flatten(input_shape=(7,7,512)))
model2.add(Dense(100, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(10, activation='softmax'))


# In[26]:


model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
model2.summary()


# In[27]:


model2.fit(vgg_features_train, train_target, epochs=50, batch_size=128, validation_data=(vgg_features_val, val_target))


# In[ ]:




