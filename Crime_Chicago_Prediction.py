#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
##import pandas_profiling
import folium
import webbrowser
from PIL import Image
#pandas_profiling.ProfileReport(USA_Housing)


# In[3]:


## Import Directory


# In[4]:


link="Dataset Location(Directory)"
dirlist=os.listdir(link)
print(dirlist)


# ## Import Data

# In[5]:


c0104=pd.read_csv(link+dirlist[1])
c0507=pd.read_csv(link+dirlist[2])
c0811=pd.read_csv(link+dirlist[3])
c1217=pd.read_csv(link+dirlist[4])
c1719=pd.read_csv(link+dirlist[6])


# In[6]:


c0104.columns


# ## Creating Data Profile

# In[80]:


#pandas_profiling.ProfileReport(c0104)


# In[81]:


#pandas_profiling.ProfileReport(c0507)


# In[82]:


#pandas_profiling.ProfileReport(c0811)


# In[83]:


#pandas_profiling.ProfileReport(c1217)


# ## Data Cleaning

# In[100]:


c0811.head()


# In[7]:



c0104smooth=c0104.dropna(axis=1)
c0104smooth.isna().sum()
c0104smooth=c0104smooth[1:]
print(len(c0104smooth))
c0104.describe().to_csv(link+"des.csv")


# In[8]:


c0507smooth=c0507.dropna(axis=1)
c0507smooth.isna().sum()
c0507smooth=c0507smooth[1:]
print(len(c0507smooth))
c0507.describe().to_csv(link+"des1.csv")


# In[108]:


c0811smooth=c0811.dropna(axis=1)
c0811smooth.isna().sum()
c0811smooth=c0811smooth[1:]
print(len(c0811smooth))
c0811.describe().to_csv(link+"des2.csv")
c0811smooth.head()


# In[10]:


c1217smooth=c1217.dropna(axis=1)
c1217smooth.isna().sum()
c1217smooth=c1217smooth[1:]
print(len(c1217smooth))
c1217.describe().to_csv(link+"des3.csv")


# ## Subsetting

# In[11]:


c1=c0104smooth.columns.tolist()
c0104smooth=c0104smooth[c1[1:]]
c0104smooth.head()


# In[12]:


c2=c0507smooth.columns.tolist()
c0507smooth=c0507smooth[c2[1:]]
c0507smooth.head()


# In[13]:


c3=c0811smooth.columns.tolist()
c0811smooth=c0811smooth[c3[1:]]
c0811smooth.head()


# In[14]:


c4=c1217smooth.columns.tolist()
c1217smooth=c1217smooth[c4[1:]]
c1217smooth.head()

c5=c1719smooth.columns.tolist()
c1719smooth=c1719smooth
c1719smooth.head()
c1719smooth.columns
# ## Primary Analysis

# In[ ]:





# In[16]:


c0104arr=np.array(c0104smooth['Primary Type'])
c0104u=np.unique(c0104arr)


# In[17]:


c0104key=np.unique(np.array(np.array(c0104smooth['Primary Type'])))
c0104val=np.array(c0104smooth['Primary Type'].value_counts().values)
print(c0104u[:8])
print(c0104val[:8])


# In[18]:


color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2001-2004 Max: {}={}".format(c0104u[:8][0],c0104val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c0104u[:8],c0104val[:8],color=color,width=0.5)


# In[19]:


c0507arr=np.array(c0507smooth['Primary Type'])
c0507u=np.unique(c0507arr)
c0507key=np.unique(np.array(np.array(c0507smooth['Primary Type'])))
c0507val=np.array(c0507smooth['Primary Type'].value_counts().values)
print(c0507u[:8])
print(c0507val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2005-2007 Max: {}={}".format(c0507u[:8][0],c0507val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c0507u[:8],c0507val[:8],color=color,width=0.5)


# In[20]:


c0811arr=np.array(c0811smooth['Primary Type'])
c0811u=np.unique(c0811arr)
c0811key=np.unique(np.array(np.array(c0811smooth['Primary Type'])))
c0811val=np.array(c0811smooth['Primary Type'].value_counts().values)
print(c0811u[:8])
print(c0811val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2008-2011 Max: {}={}".format(c0811u[:8][0],c0811val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c0811u[:8],c0811val[:8],color=color,width=0.5)


# In[21]:


c1217arr=np.array(c1217smooth['Primary Type'])
c1217u=np.unique(c1217arr)
c1217key=np.unique(np.array(np.array(c1217smooth['Primary Type'])))
c1217val=np.array(c1217smooth['Primary Type'].value_counts().values)
print(c1217u[:8])
print(c1217val[:8])
color='cmykrgb'
plt.figure(figsize=(30,9))
plt.title("Crime Scenario of 2012-2017 Max: {}={}".format(c1217u[:8][0],c1217val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c1217u[:8],c1217val[:8],color=color,width=0.5)


# In[47]:


for i in range(len(c0104val[:8])):
    val=[c0104val[:8][i],c0507val[:8][i],c0811val[:8][i],c1217val[:8][i]]
    year=['2001-2004','2005-2007','2008-2011','2012-2017']
    color='cmykrgb'
    plt.figure(figsize=(18,7))
    plt.title("Crime Scenario of {}".format(c1217u[:8][i]),fontsize=20,color='m')
    plt.xlabel("Crime Year->",fontsize=15,color='b')
    plt.ylabel("Crime Occured->",fontsize=15,color='b')
    plt.grid()
    plt.bar(year,val,color=color,width=0.5)
    plt.savefig("C:/Users/maiti/OneDrive/Desktop/crimes-in-chicago/viz/"+c1217u[:8][i])


# ### 2001

# In[24]:


c01=c0104smooth[c0104smooth['Year']==2001]
c01=c01[c01.columns.tolist()[1:]]
c01.head()
c01arr=np.array(c01['Primary Type'])
c01u=np.unique(c01arr)
c01key=np.unique(np.array(np.array(c01['Primary Type'])))
c01val=np.array(c01['Primary Type'].value_counts().values)
print(c01u[:8])
print(c01val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2001 Max: {}={}".format(c01u[:8][0],c01val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c01u[:8],c01val[:8],color=color,width=0.5)


# ### 2002

# In[25]:


c02=c0104smooth[c0104smooth['Year']==2002]
c02=c02[c02.columns.tolist()[1:]]
c02.head()
c02arr=np.array(c01['Primary Type'])
c02u=np.unique(c02arr)
c02key=np.unique(np.array(np.array(c02['Primary Type'])))
c02val=np.array(c02['Primary Type'].value_counts().values)
print(c02u[:8])
print(c02val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2002 Max: {}={}".format(c02u[:8][0],c02val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c02u[:8],c02val[:8],color=color,width=0.5)


# ### 2003

# In[26]:


c03=c0104smooth[c0104smooth['Year']==2003]
c03=c03[c03.columns.tolist()[1:]]
c03.head()
c03arr=np.array(c03['Primary Type'])
c03u=np.unique(c03arr)
c03key=np.unique(np.array(np.array(c03['Primary Type'])))
c03val=np.array(c03['Primary Type'].value_counts().values)
print(c03u[:8])
print(c03val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2003 Max: {}={}".format(c03u[:8][0],c03val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c03u[:8],c03val[:8],color=color,width=0.5)


# ### 2004

# In[27]:


c04=c0104smooth[c0104smooth['Year']==2004]
c04=c04[c04.columns.tolist()[1:]]
c04.head()
c04arr=np.array(c04['Primary Type'])
c04u=np.unique(c04arr)
c04key=np.unique(np.array(np.array(c04['Primary Type'])))
c04val=np.array(c04['Primary Type'].value_counts().values)
print(c04u[:8])
print(c04val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2004 Max: {}={}".format(c04u[:8][0],c04val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c04u[:8],c04val[:8],color=color,width=0.5)


# ### 2005

# In[28]:


c05=c0507smooth[c0507smooth['Year']==2005]
c05=c05[c05.columns.tolist()[1:]]
c05.head()
c05arr=np.array(c05['Primary Type'])
c05u=np.unique(c05arr)
c05key=np.unique(np.array(np.array(c05['Primary Type'])))
c05val=np.array(c05['Primary Type'].value_counts().values)
print(c05u[:8])
print(c05val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2005 Max: {}={}".format(c05u[:8][0],c05val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c05u[:8],c05val[:8],color=color,width=0.5)


# ### 2006

# In[29]:


c06=c0507smooth[c0507smooth['Year']==2006]
c06=c06[c06.columns.tolist()[1:]]
c06.head()
c06arr=np.array(c06['Primary Type'])
c06u=np.unique(c06arr)
c06key=np.unique(np.array(np.array(c06['Primary Type'])))
c06val=np.array(c06['Primary Type'].value_counts().values)
print(c06u[:8])
print(c06val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2006 Max: {}={}".format(c06u[:8][0],c06val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c06u[:8],c06val[:8],color=color,width=0.5)


# ### 2007

# In[30]:


c07=c0507smooth[c0507smooth['Year']==2007]
c07=c07[c07.columns.tolist()[1:]]
c07.head()
c07arr=np.array(c07['Primary Type'])
c07u=np.unique(c07arr)
c07key=np.unique(np.array(np.array(c07['Primary Type'])))
c07val=np.array(c07['Primary Type'].value_counts().values)
print(c07u[:8])
print(c07val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2007 Max: {}={}".format(c07u[:8][0],c07val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c07u[:8],c07val[:8],color=color,width=0.5)


# ### 2008

# In[106]:


c08=c0811smooth[c0811smooth['Year']==2008]
c08=c08[c08.columns.tolist()[1:]]
c08.head()
c08arr=np.array(c08['Primary Type'])
c08u=np.unique(c08arr)
c08key=np.unique(np.array(np.array(c08['Primary Type'])))
c08val=np.array(c08['Primary Type'].value_counts().values)
print(c08u[:8])
print(c08val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2008 Max: {}={}".format(c08u[:8][0],c08val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c08u[:8],c08val[:8],color=color,width=0.5)
c08.head()


# ### 2009

# In[32]:


c09=c0811smooth[c0811smooth['Year']==2009]
c09=c09[c09.columns.tolist()[1:]]
c09.head()
c09arr=np.array(c09['Primary Type'])
c09u=np.unique(c09arr)
c09key=np.unique(np.array(np.array(c09['Primary Type'])))
c09val=np.array(c09['Primary Type'].value_counts().values)
print(c09u[:8])
print(c09val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2009 Max: {}={}".format(c09u[:8][0],c09val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c09u[:8],c09val[:8],color=color,width=0.5)


# ### 2010

# In[33]:


c10=c0811smooth[c0811smooth['Year']==2010]
c10=c10[c10.columns.tolist()[1:]]
c10.head()
c10arr=np.array(c10['Primary Type'])
c10u=np.unique(c10arr)
c10key=np.unique(np.array(np.array(c10['Primary Type'])))
c10val=np.array(c10['Primary Type'].value_counts().values)
print(c10u[:8])
print(c10val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2010 Max: {}={}".format(c10u[:8][0],c10val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c10u[:8],c10val[:8],color=color,width=0.5)


# ### 2011

# In[34]:


c11=c0811smooth[c0811smooth['Year']==2011]
c11=c11[c11.columns.tolist()[1:]]
c11.head()
c11arr=np.array(c11['Primary Type'])
c11u=np.unique(c11arr)
c11key=np.unique(np.array(np.array(c11['Primary Type'])))
c11val=np.array(c11['Primary Type'].value_counts().values)
print(c11u[:8])
print(c11val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2011 Max: {}={}".format(c11u[:8][0],c11val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c11u[:8],c11val[:8],color=color,width=0.5)


# ### 2012

# In[35]:


c12=c1217smooth[c1217smooth['Year']==2012]
c12=c12[c12.columns.tolist()[1:]]
c12.head()
c12arr=np.array(c12['Primary Type'])
c12u=np.unique(c12arr)
c12key=np.unique(np.array(np.array(c12['Primary Type'])))
c12val=np.array(c12['Primary Type'].value_counts().values)
print(c12u[:8])
print(c12val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2012 Max: {}={}".format(c12u[:8][0],c12val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c12u[:8],c12val[:8],color=color,width=0.5)


# ### 2013

# In[36]:


c13=c1217smooth[c1217smooth['Year']==2013]
c13=c13[c13.columns.tolist()[1:]]
c13.head()
c13arr=np.array(c13['Primary Type'])
c13u=np.unique(c13arr)
c13key=np.unique(np.array(np.array(c13['Primary Type'])))
c13val=np.array(c13['Primary Type'].value_counts().values)
print(c13u[:8])
print(c13val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2013 Max: {}={}".format(c13u[:8][0],c13val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c13u[:8],c13val[:8],color=color,width=0.5)


# ### 2014

# In[37]:


c14=c1217smooth[c1217smooth['Year']==2014]
c14=c14[c14.columns.tolist()[1:]]
c14.head()
c14arr=np.array(c14['Primary Type'])
c14u=np.unique(c14arr)
c14key=np.unique(np.array(np.array(c14['Primary Type'])))
c14val=np.array(c14['Primary Type'].value_counts().values)
print(c14u[:8])
print(c14val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2014 Max: {}={}".format(c14u[:8][0],c14val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c14u[:8],c14val[:8],color=color,width=0.5)


# ### 2015

# In[38]:


c15=c1217smooth[c1217smooth['Year']==2015]
c15=c15[c15.columns.tolist()[1:]]
c15.head()
c15arr=np.array(c15['Primary Type'])
c15u=np.unique(c15arr)
c15key=np.unique(np.array(np.array(c15['Primary Type'])))
c15val=np.array(c15['Primary Type'].value_counts().values)
print(c15u[:8])
print(c15val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2015 Max: {}={}".format(c15u[:8][0],c15val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=15,color='b')
plt.ylabel("Crime Occured->",fontsize=15,color='b')
plt.grid()
plt.bar(c15u[:8],c15val[:8],color=color,width=0.5)


# ### 2016

# In[39]:


c16=c1217smooth[c1217smooth['Year']==2016]
c16=c16[c16.columns.tolist()[1:]]
c16.head()
c16arr=np.array(c16['Primary Type'])
c16u=np.unique(c16arr)
c16key=np.unique(np.array(np.array(c16['Primary Type'])))
c16val=np.array(c16['Primary Type'].value_counts().values)
print(c16u[:8])
print(c16val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2016 Max: {}={}".format(c16u[:8][0],c16val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=16,color='b')
plt.ylabel("Crime Occured->",fontsize=16,color='b')
plt.grid()
plt.bar(c16u[:8],c16val[:8],color=color,width=0.5)


# ### 2017

# In[40]:


c17=c0811smooth[c0811smooth['Year']==2009]
c17=c17[c17.columns.tolist()[1:]]
c17.head()
c17arr=np.array(c17['Primary Type'])
c17u=np.unique(c17arr)
c17key=np.unique(np.array(np.array(c17['Primary Type'])))
c17val=np.array(c17['Primary Type'].value_counts().values)
print(c17u[:8])
print(c17val[:8])
color='cmykrgb'
plt.figure(figsize=(20,7))
plt.title("Crime Scenario of 2017 Max: {}={}".format(c17u[:8][0],c17val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=17,color='b')
plt.ylabel("Crime Occured->",fontsize=17,color='b')
plt.grid()
plt.bar(c17u[:8],c17val[:8],color=color,width=0.5)
plt.figure(figsize=(18,10))
plt.title("Crime Scenario of 2017",fontsize=20,color='m')
plt.ylabel("Crime Type->",fontsize=17,color='b')
plt.xlabel("Crime Occured->",fontsize=17,color='b')
plt.grid()
plt.barh(c17u,c17val)
#c17.groupby('Primary Type')['Arrest'].count()


# ### 2018

# In[41]:


c18=c1719[c1719['Year']==2018]
c18=c18[c18.columns.tolist()[1:]]
c18.head()
c18arr=np.array(c18['Primary Type'])
c18u=np.unique(c18arr)
c18key=np.unique(np.array(np.array(c18['Primary Type'])))
c18val=np.array(c18['Primary Type'].value_counts().values)
print(c18u[:8])
print(c18val[:8])
color='cmykrgb'
plt.figure(figsize=(28,7))
plt.title("Crime Scenario of 2018 Max: {}={}".format(c18u[:8][0],c18val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=18,color='b')
plt.ylabel("Crime Occured->",fontsize=18,color='b')
plt.grid()
plt.bar(c18u[:8],c18val[:8],color=color,width=0.5)

plt.figure(figsize=(18,10))
plt.title("Crime Scenario of 2018",fontsize=20,color='m')
plt.ylabel("Crime Type->",fontsize=17,color='b')
plt.xlabel("Crime Occured->",fontsize=17,color='b')
plt.grid()
plt.barh(c18u,c18val)
c18.head()


# ### 2019 

# In[42]:


c19=c1719[c1719['Year']==2019]
c19=c19[c19.columns.tolist()[1:]]
c19.head()
c19arr=np.array(c19['Primary Type'])
c19u=np.unique(c19arr)
c19key=np.unique(np.array(np.array(c19['Primary Type'])))
c19val=np.array(c19['Primary Type'].value_counts().values)
print(c19u[:8])
print(c19val[:8])
color='cmykrgb'
plt.figure(figsize=(28,7))
plt.title("Crime Scenario of 2019 Max: {}={}".format(c19u[:8][0],c19val[:8][0]),fontsize=20,color='m')
plt.xlabel("Crime Type->",fontsize=19,color='b')
plt.ylabel("Crime Occured->",fontsize=19,color='b')
plt.grid()
plt.bar(c19u[:8],c19val[:8],color=color,width=0.5)

plt.figure(figsize=(18,10))
plt.title("Crime Scenario of 2019",fontsize=20,color='m')
plt.ylabel("Crime Type->",fontsize=17,color='b')
plt.xlabel("Crime Occured->",fontsize=17,color='b')
plt.grid()
plt.barh(c19u,c19val)


# In[43]:


crmyr=['2019','2018','2017','2016','2015','2014','2013','2012']
crm=[c19val[:8][0],c18val[:8][0],c17val[:8][0],c16val[:8][0],c15val[:8][0],c14val[:8][0],c13val[:8][0],c12val[:8][0]]


# In[48]:


for i in range(len(c19u[:8])):
    plt.figure(figsize=(10,5))
    crmyr=['2019','2018','2017','2016','2015','2014','2013','2012']
    crm=[c19val[:8][i],c18val[:8][i],c17val[:8][i],c16val[:8][i],c15val[:8][i],c14val[:8][i],c13val[:8][i],c12val[:8][i]]
    plt.title("{} Crime Rate".format(c19u[:8][i]),fontsize=20,color="m")
    plt.xlabel("Year->",fontsize=13,color="b")
    plt.ylabel("Count of Crime",fontsize=13,color="b")
    plt.grid()
    plt.plot(crmyr[::-1],crm[::-1],"g")
    plt.plot(crmyr[::-1],crm[::-1],"Dr")
    plt.savefig("Directory Location"+c19u[:8][i]+"_graph")


# ## Comparative analysis of 19 years

# #### Arrest

# In[52]:


c0104arst=np.array(c0104['Arrest'])
c0104arstu=np.unique(c0104arst)
print(c0104arstu)
c0104arst=c0104arst.tolist()
#print(c19arst)
cntt0104=c0104arst.count(c0104arstu[0])
cntf0104=c0104arst.count(c0104arstu[1])
cnt0104=[cntt0104,cntf0104]
print(cnt0104)


# In[78]:


arrst=[]
arrstratf=[]
arrstratt=[]


# #### Arrest in 2001

# In[79]:


c01arrststs=c01.groupby(['Arrest']).count()['Case Number'].keys().tolist()
c01arrstcnt=c01.groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c01arrststs)
print(c01arrstcnt)


# In[80]:


arrst.append(c01arrstcnt[0])
arrstratf.append((c01arrstcnt[0]/sum(c01arrstcnt))*100)
arrstratt.append((c01arrstcnt[1]/sum(c01arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2002

# In[82]:


c02arrststs=c02.groupby(['Arrest']).count()['Case Number'].keys().tolist()
c02arrstcnt=c02.groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c02arrststs)
print(c02arrstcnt)


# In[83]:


arrst.append(c02arrstcnt[0])
arrstratf.append((c02arrstcnt[0]/sum(c02arrstcnt))*100)
arrstratt.append((c02arrstcnt[1]/sum(c02arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2003

# In[84]:


c03arrststs=c03.groupby(['Arrest']).count()['Case Number'].keys().tolist()
c03arrstcnt=c03.groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c03arrststs)
print(c03arrstcnt)


# In[85]:


arrst.append(c03arrstcnt[0])
arrstratf.append((c03arrstcnt[0]/sum(c03arrstcnt))*100)
arrstratt.append((c03arrstcnt[1]/sum(c03arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2004

# In[88]:


c04arrststs=c04.groupby(['Arrest']).count()['Case Number'].keys().tolist()
c04arrstcnt=c04.groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c04arrststs)
print(c04arrstcnt)


# In[89]:


arrst.append(c04arrstcnt[0])
arrstratf.append((c04arrstcnt[0]/sum(c04arrstcnt))*100)
arrstratt.append((c04arrstcnt[1]/sum(c04arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2005

# In[90]:


c05arrststs=c05.groupby(['Arrest']).count()['Case Number'].keys().tolist()
c05arrstcnt=c05.groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c05arrststs)
print(c05arrstcnt)


# In[91]:


arrst.append(c05arrstcnt[0])
arrstratf.append((c05arrstcnt[0]/sum(c05arrstcnt))*100)
arrstratt.append((c05arrstcnt[1]/sum(c05arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2006

# In[92]:


c06arrststs=c06.groupby(['Arrest']).count()['Case Number'].keys().tolist()
c06arrstcnt=c06.groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c06arrststs)
print(c06arrstcnt)


# In[93]:


arrst.append(c06arrstcnt[0])
arrstratf.append((c06arrstcnt[0]/sum(c06arrstcnt))*100)
arrstratt.append((c06arrstcnt[1]/sum(c06arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2007

# In[94]:


c07arrststs=c07.groupby(['Arrest']).count()['Case Number'].keys().tolist()
c07arrstcnt=c07.groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c07arrststs)
print(c07arrstcnt)


# In[95]:


arrst.append(c07arrstcnt[0])
arrstratf.append((c07arrstcnt[0]/sum(c07arrstcnt))*100)
arrstratt.append((c07arrstcnt[1]/sum(c07arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2008

# In[103]:


c08.columns.tolist()


# In[109]:


c08arrststs=c0811[c0811['Year']==2008].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c08arrstcnt=c0811[c0811['Year']==2008].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c08arrststs)
print(c08arrstcnt)


# In[110]:


arrst.append(c08arrstcnt[0])
arrstratf.append((c08arrstcnt[0]/sum(c08arrstcnt))*100)
arrstratt.append((c08arrstcnt[1]/sum(c08arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2009

# In[111]:


c09arrststs=c0811[c0811['Year']==2009].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c09arrstcnt=c0811[c0811['Year']==2009].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c09arrststs)
print(c09arrstcnt)


# In[112]:


arrst.append(c09arrstcnt[0])
arrstratf.append((c09arrstcnt[0]/sum(c09arrstcnt))*100)
arrstratt.append((c09arrstcnt[1]/sum(c09arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2010

# In[113]:


c10arrststs=c0811[c0811['Year']==2010].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c10arrstcnt=c0811[c0811['Year']==2010].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c10arrststs)
print(c10arrstcnt)


# In[114]:


arrst.append(c10arrstcnt[0])
arrstratf.append((c10arrstcnt[0]/sum(c10arrstcnt))*100)
arrstratt.append((c10arrstcnt[1]/sum(c10arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2011

# In[115]:


c11arrststs=c0811[c0811['Year']==2011].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c11arrstcnt=c0811[c0811['Year']==2011].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c11arrststs)
print(c11arrstcnt)


# In[116]:


arrst.append(c11arrstcnt[0])
arrstratf.append((c11arrstcnt[0]/sum(c11arrstcnt))*100)
arrstratt.append((c11arrstcnt[1]/sum(c11arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2012

# In[117]:


c12arrststs=c1217[c1217['Year']==2012].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c12arrstcnt=c1217[c1217['Year']==2012].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c12arrststs)
print(c12arrstcnt)


# In[118]:


arrst.append(c12arrstcnt[0])
arrstratf.append((c12arrstcnt[0]/sum(c12arrstcnt))*100)
arrstratt.append((c12arrstcnt[1]/sum(c12arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2013

# In[119]:


c13arrststs=c1217[c1217['Year']==2013].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c13arrstcnt=c1217[c1217['Year']==2013].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c13arrststs)
print(c13arrstcnt)


# In[120]:


arrst.append(c13arrstcnt[0])
arrstratf.append((c13arrstcnt[0]/sum(c13arrstcnt))*100)
arrstratt.append((c13arrstcnt[1]/sum(c13arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2014

# In[121]:


c14arrststs=c1217[c1217['Year']==2014].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c14arrstcnt=c1217[c1217['Year']==2014].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c14arrststs)
print(c14arrstcnt)


# In[122]:


arrst.append(c14arrstcnt[0])
arrstratf.append((c14arrstcnt[0]/sum(c14arrstcnt))*100)
arrstratt.append((c14arrstcnt[1]/sum(c14arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2015

# In[123]:


c15arrststs=c1217[c1217['Year']==2015].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c15arrstcnt=c1217[c1217['Year']==2015].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c15arrststs)
print(c15arrstcnt)


# In[124]:


arrst.append(c15arrstcnt[0])
arrstratf.append((c15arrstcnt[0]/sum(c15arrstcnt))*100)
arrstratt.append((c15arrstcnt[1]/sum(c15arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2016

# In[125]:


c16arrststs=c1217[c1217['Year']==2016].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c16arrstcnt=c1217[c1217['Year']==2016].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c16arrststs)
print(c16arrstcnt)


# In[126]:


arrst.append(c16arrstcnt[0])
arrstratf.append((c16arrstcnt[0]/sum(c16arrstcnt))*100)
arrstratt.append((c16arrstcnt[1]/sum(c16arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2017

# In[129]:


c17arrststs=c1719[c1719['Year']==2017].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c17arrstcnt=c1719[c1719['Year']==2017].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c17arrststs)
print(c17arrstcnt)


# In[130]:


arrst.append(c17arrstcnt[0])
arrstratf.append((c17arrstcnt[0]/sum(c17arrstcnt))*100)
arrstratt.append((c17arrstcnt[1]/sum(c17arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2018

# In[131]:


c18arrststs=c1719[c1719['Year']==2018].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c18arrstcnt=c1719[c1719['Year']==2018].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c18arrststs)
print(c18arrstcnt)


# In[132]:


arrst.append(c18arrstcnt[0])
arrstratf.append((c18arrstcnt[0]/sum(c18arrstcnt))*100)
arrstratt.append((c18arrstcnt[1]/sum(c18arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# #### Arrest in 2019

# In[133]:


c19arrststs=c1719[c1719['Year']==2019].groupby(['Arrest']).count()['Case Number'].keys().tolist()
c19arrstcnt=c1719[c1719['Year']==2019].groupby(['Arrest']).count()['Case Number'].values.tolist()
print(c19arrststs)
print(c19arrstcnt)


# In[134]:


arrst.append(c19arrstcnt[0])
arrstratf.append((c19arrstcnt[0]/sum(c19arrstcnt))*100)
arrstratt.append((c19arrstcnt[1]/sum(c19arrstcnt))*100)
print(arrst)
print(arrstratf)
print(arrstratt)


# In[135]:


print(len(arrst))
print(len(arrstratf))
print(len(arrstratt))


# In[141]:


yearlist=[str(i) for i in range(2001,2020)]
print(yearlist)


# In[157]:


plt.figure(figsize=(15,6))
plt.title("Crime Arrest Report",fontsize=20,color='m')
plt.xlabel("Year->",fontsize=15,color='b')
plt.ylabel("Arrest percentage->",fontsize=15,color='b')
plt.plot(arrstratt,"k",label="Arrested")
plt.plot(arrstratt,"Pk")
plt.plot(arrstratf,"r",label="Not Arrested")
plt.plot(arrstratf,"Dr")
plt.legend(loc="upper right")


# In[158]:


plt.figure(figsize=(15,6))
plt.title("Crime Arrest Report Yearwise",fontsize=20,color='m')
plt.xlabel("Year->",fontsize=15,color='b')
plt.ylabel("Arrest percentage->",fontsize=15,color='b')
plt.bar(yearlist,arrst,color="crmgybk")


# In[165]:


arrsted=[c01arrstcnt[1],c02arrstcnt[1],c03arrstcnt[1],c04arrstcnt[1],c05arrstcnt[1],c06arrstcnt[1],c07arrstcnt[1],c08arrstcnt[1],c09arrstcnt[1],c10arrstcnt[1],c11arrstcnt[1],c12arrstcnt[1],c13arrstcnt[1],c14arrstcnt[1],c15arrstcnt[1],c16arrstcnt[1],c17arrstcnt[1],c18arrstcnt[1],c19arrstcnt[1]]
print(len(arrsted))


# In[173]:


prob=[]
pres=arrstratt[0]
for i in range(len(arrstratt)):
    if pres>=arrstratt[i]:
        prob.append(0)
    else:
        prob.append(1)
    pres=arrstratt[i]
print(prob)


# In[193]:


ratio=[]
for i in range(len(arrstratt)):
    ratio.append(arrsted[i]/(arrst[i]+arrsted[i]))
print(ratio)


# In[194]:


regdata=pd.DataFrame({
    "Year":yearlist,
    "Case Booked":np.array(arrst)+np.array(arrsted),
    "Not Arrested":arrst,
    "Arrested":arrsted,
    "Arrested(%)":arrstratt,
    "Not Arrested(%)":arrstratf,
    "Efficiency Ratio":ratio,
    "Crime Hike":prob
})

regdata.to_csv("C:/Users/maiti/OneDrive/Desktop/crimes-in-chicago/regdata.csv")


# In[188]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(regdata[regdata.columns.tolist()[1:len(regdata.columns.tolist())-1]],regdata[regdata.columns.tolist()[-1]])


# In[189]:


regdata.columns.tolist()


# In[197]:


bookcase=int(input("Enter Probable Book Case number: "))
meanrat=np.mean(np.array(regdata['Efficiency Ratio']))
predict_crime=logmodel.predict([[bookcase,bookcase-bookcase*meanrat,bookcase*meanrat,bookcase*meanrat/bookcase,bookcase-(bookcase*meanrat/bookcase),meanrat]])
print(predict_crime[0])


# In[198]:


if predict_crime[0]==0:
    print("Crime in Chicago will be Decreased...")
else:
    print("Crime in Chicago will be Increased...")


# In[195]:


regdata.head()


# In[168]:


regdata.corr()


# In[124]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2019")
sns.countplot(c19['Arrest'])


# In[125]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2018")
sns.countplot(c18['Arrest'])


# In[126]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2017")
sns.countplot(c17['Arrest'])


# In[127]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2016")
sns.countplot(c16['Arrest'])


# In[128]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2015")
sns.countplot(c15['Arrest'])


# In[129]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2014")
sns.countplot(c14['Arrest'])


# In[130]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2013")
sns.countplot(c13['Arrest'])


# In[131]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2012")
sns.countplot(c12['Arrest'])


# In[132]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2011")
sns.countplot(c11['Arrest'])


# In[133]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2010")
sns.countplot(c10['Arrest'])


# In[134]:


plt.figure(figsize=(8,6))
plt.title("Arrest in 2009")
sns.countplot(c09['Arrest'])


# ## Primary Type with Arrest

# In[135]:


np.array(c19.groupby('Primary Type')['Arrest'].count())


# In[136]:


np.array(c18.groupby('Primary Type')['Arrest'].count())


# In[137]:


np.array(c17.groupby('Primary Type')['Arrest'].count())


# In[138]:


np.array(c16.groupby('Primary Type')['Arrest'].count())


# In[139]:


np.array(c15.groupby('Primary Type')['Arrest'].count())


# In[140]:


np.array(c14.groupby('Primary Type')['Arrest'].count())


# In[141]:


np.array(c13.groupby('Primary Type')['Arrest'].count())


# In[142]:


np.array(c12.groupby('Primary Type')['Arrest'].count())


# In[143]:


np.array(c11.groupby('Primary Type')['Arrest'].count())


# In[144]:


np.array(c10.groupby('Primary Type')['Arrest'].count())


# In[145]:


np.array(c09.groupby('Primary Type')['Arrest'].count())


# In[ ]:





# In[146]:


c19num=c19[['Beat','District','Ward','Community Area']]
sns.heatmap(c19num.corr(),annot=True)


# In[147]:


c18num=c18[['Beat','District','Ward','Community Area']]
sns.heatmap(c18num.corr(),annot=True)


# In[148]:


c01num=c01[['Beat','District']]
sns.heatmap(c01num.corr(),annot=True)


# In[158]:


c19.corr().to_csv(link+"corr.csv")


# In[149]:


plt.figure(figsize=(15,15))
sns.heatmap(c19.corr(),annot=True)


# In[168]:


c18.head()


# In[155]:


c19['Latitude'].iloc[0]


# In[156]:


import folium 
import folium.plugins as plugins
import branca


# In[174]:


'''18=c18.fillna(0)'''
'''print(c18)
print(c18['Latitude'].iloc[0])
locations = c18['Latitude'].iloc[0], c18['Longitude'].iloc[0]'''
print("Lat Mean: ",c18['Latitude'].iloc[0])
print("Long Mean: ",c18smooth['Longitude'].iloc[0])
m = folium.Map(location=locations,zoom_start=7)
'''c18=c18.fillna(0,inplace=True)'''
print(type(c18))
def colsel(i):
    if c18['Arrest'].iloc[i] == 1:
        color = 'blue'
    elif c18['Arrest'].iloc[i] == 2:
        color = 'green'
    else:
        color = 'red'
    return color

for i in range(100):
    #m = folium.Map(location=location,zoom_start=12)
    #print(accupdate.iloc[i]['Latitude']," ", accupdate.iloc[i]['Longitude'])
    show="Accident Loc=> Lat:"+str(c18.iloc[i]['Latitude'])+" long:"+str(c18.iloc[i]['Longitude'])
    popup = folium.Popup(show, parse_html=True) 
    folium.Marker([c18.iloc[i]['Latitude'], c18.iloc[i]['Longitude']],popup=popup,
                  icon=folium.Icon(color=colsel(i))
                 ).add_to(m)
    #m.add_child(folium.Marker(locations,popup="Accident Loc=> Lat:"+str(accupdate.iloc[i]['Latitude'])+" long:"+str(accupdate.iloc[i]['Longitude']),icon=folium.Icon(color="blue", icon='info-sign')))
m.save('D:/As Freelancer/Live Assignment/DigiVersal/DVMAY004/Accident_Severity2.html')


# In[ ]:





# In[ ]:




