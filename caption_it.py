#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model


# In[3]:


model = load_model('./model_weights/model_9_new.h5')


# In[4]:


model_temp = InceptionV3(weights='imagenet')


# In[5]:


new_model = Model(model_temp.input, model_temp.layers[-2].output)


# In[6]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
def preprocess(image_path):
    images = image.load_img(image_path, target_size = (299, 299))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# In[7]:


def encode(image):
    image = preprocess(image)
    feature_vector = new_model.predict(image)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector


# In[8]:


import pickle

with open('./storage/wordtoix.pkl', 'rb') as w2i:
    wordtoix = pickle.load(w2i)
    
with open('./storage/ixtoword.pkl', 'rb') as i2w:
    ixtoword = pickle.load(i2w)


# In[9]:


def Caption_Photo(photo):
    in_text = 'startseq'
    max_lengths = 35
    for i in range(max_lengths):
        sequence = [wordtoix[word] for word in in_text.split() if word in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_lengths)
        y_pred = model.predict([photo,sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = ixtoword[y_pred]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[10]:


def caption_this_image(image):
    enc = encode(image)
    enc = enc.reshape(1, 2048)
    caption = Caption_Photo(enc)
    return caption


# In[ ]:




