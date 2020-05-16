#!/usr/bin/env python
# coding: utf-8

# # Taxt to Speech converter By Karan Patel

#  How can you use this program ?
#  
#  1. Simple taxt reader (One line reader)
#  
#  2. You can make text reading application like (Amazon Kindle, google book reader)
#  
#  3. Assistive device for blind (Bio medical application)

# In[12]:


# Import the required module for text 
# to speech conversion 

from pygame import mixer
from gtts import gTTS
import os
import sys
import msvcrt


# In[8]:


text_en  = input("write here what ever you want to convert into speak :         ")


# In[3]:


# Language in which you want to convert 
language = 'hi'

## Supported Languages <a name="lang_list"></a>


# * 'af' : 'Afrikaans'
# * 'sq' : 'Albanian'
# * 'ar' : 'Arabic'
# * 'hy' : 'Armenian'
# * 'bn' : 'Bengali'
# * 'ca' : 'Catalan'
# * 'zh' : 'Chinese'
# * 'zh-cn' : 'Chinese (Mandarin/China)'
# * 'zh-tw' : 'Chinese (Mandarin/Taiwan)'
# * 'zh-yue' : 'Chinese (Cantonese)'
# * 'hr' : 'Croatian'
# * 'cs' : 'Czech'
# * 'da' : 'Danish'
# * 'nl' : 'Dutch'
# * 'en' : 'English'
# * 'en-au' : 'English (Australia)'
# * 'en-uk' : 'English (United Kingdom)'
# * 'en-us' : 'English (United States)'
# * 'eo' : 'Esperanto'
# * 'fi' : 'Finnish'
# * 'fr' : 'French'
# * 'de' : 'German'
# * 'el' : 'Greek'
# * 'hi' : 'Hindi'
# * 'hu' : 'Hungarian'
# * 'is' : 'Icelandic'
# * 'id' : 'Indonesian'
# * 'it' : 'Italian'
# * 'ja' : 'Japanese'
# * 'km' : 'Khmer (Cambodian)'
# * 'ko' : 'Korean'
# * 'la' : 'Latin'
# * 'lv' : 'Latvian'
# * 'mk' : 'Macedonian'
# * 'no' : 'Norwegian'
# * 'pl' : 'Polish'
# * 'pt' : 'Portuguese'
# * 'ro' : 'Romanian'
# * 'ru' : 'Russian'
# * 'sr' : 'Serbian'
# * 'si' : 'Sinhala'
# * 'sk' : 'Slovak'
# * 'es' : 'Spanish'
# * 'es-es' : 'Spanish (Spain)'
# * 'es-us' : 'Spanish (United States)'
# * 'sw' : 'Swahili'
# * 'sv' : 'Swedish'
# * 'ta' : 'Tamil'
# * 'th' : 'Thai'
# * 'tr' : 'Turkish'
# * 'uk' : 'Ukrainian'
# * 'vi' : 'Vietnamese'
# * 'cy' : 'Welsh'

# In[9]:


myobj = gTTS(text=text_en, lang=language, slow=False) 


# In[10]:


# Saving the converted audio in a mp3 file named 

myobj.save("karan_hindi.mp3") 


# In[11]:


# Playing the converted file

mixer.init()

mixer.music.load("karan_hindi.mp3")

mixer.music.play()


# In[ ]:





# In[ ]:




