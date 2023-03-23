#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


salary_data=pd.DataFrame({'salary':[900,3100,2500,5000],
                          'range':['low','mid','high','ultra']})


# In[5]:


salary_data


# In[9]:


a=salary_data.range.map({'low':1,'mid':2,'high':3,'ultra':4})


# In[10]:


a


# In[15]:


a=salary_data.salary.map({900:1,3100:2,2500:3,5000:4})


# In[16]:


a


# # Normalization

# In[1]:


pip install matplotlib


# In[1]:


pip install scikit-learn


# In[3]:


import matplotlib.pyplot as plt


# In[6]:


from mpl_toolkits.mplot3d import Axes3D


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import numpy as np


# In[9]:


import pandas as pd


# In[10]:


normalization_df=pd.DataFrame({'x':np.random.randint(-100,100,1000),
                              'y':np.random.randint(-80,80,1000),
                              'z':np.random.randint(-150,150,1000)})


# In[11]:


normalization_df


# In[15]:


ax=plt.axes(projection='3d')
ax.scatter3D(normalization_df.x,normalization_df.y,
            normalization_df.z)
plt.figure()


# In[16]:


from sklearn.preprocessing import Normalizer


# In[22]:


normal=Normalizer()


# In[23]:


normalization_df=normal.fit_transform(normalization_df)


# In[26]:


norm_df=pd.DataFrame(normalization_df,columns=['x1','x2','x3'])


# In[27]:


norm_df


# In[28]:


ax=plt.axes(projection='3d')
ax.scatter3D(norm_df.x1,norm_df.x2,
            norm_df.x3)
plt.figure()


# # DATA SORTING

# In[42]:


data=[10,11,12,18,25,36,55,11,22,25,50,89,15,65,48,24]


# In[43]:


data.sort()
print(data)


# In[46]:


data=[10,11,12,18,25,36,55,11,22,25,50,89,15,65,48,24]


# In[47]:


data.reverse()
print(data)


# # MISSING VALUES

# In[48]:


import numpy as np


# In[49]:


import pandas as pd


# In[53]:


data={'first score':[100,90,np.nan,94],
     'second score':[30,np.nan,45,56],
     'third score':[40,80,98,np.nan]}


# In[57]:


scores=pd.DataFrame(data)


# In[58]:


scores


# In[59]:


scores.isnull()


# In[61]:


from sklearn.impute import SimpleImputer


# In[62]:


imputer=SimpleImputer(missing_values=np.nan,
                     strategy='median')


# In[63]:


imputer=SimpleImputer(missing_values=np.nan,
                     strategy='mean')


# In[64]:


imputer=SimpleImputer(missing_values=np.nan,
                     strategy='mode')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




