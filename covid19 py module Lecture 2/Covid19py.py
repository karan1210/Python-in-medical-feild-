#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:


import COVID19Py


# In[2]:


covid19 = COVID19Py.COVID19()


# In[ ]:


latest = covid19.getLatest()


# In[ ]:


# Expected one of: ['confirmed', 'deaths', 'recovered']
locations = covid19.getLocations(rank_by='recovered')


# In[ ]:


print(locations)


# In[ ]:


death = covid19.getLocations(rank_by='deaths')


# In[ ]:


print (death)


# In[3]:


location = covid19.getLocationByCountryCode("India", timelines=True)


# In[ ]:




