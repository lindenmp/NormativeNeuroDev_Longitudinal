#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[1]:


# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/neurodev_long/code/func/')
from proj_environment import set_proj_env
from func import update_progress


# In[3]:


exclude_str = 't1Exclude'
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str)


# ### Setup output directory

# In[4]:


print(os.environ['MODELDIR'])
if not os.path.exists(os.environ['MODELDIR']): os.makedirs(os.environ['MODELDIR'])


# ## Load train/test .csv and setup node .csv

# In[5]:


df = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_pheno.csv'))
df.set_index(['bblid', 'scanid', 'timepoint'], inplace = True)

print(df.shape)


# In[6]:


df.head()


# In[7]:


metrics = ('ct', 'vol')


# In[8]:


# output dataframe
ct_labels = ['ct_' + str(i) for i in range(num_parcels)]
vol_labels = ['vol_' + str(i) for i in range(num_parcels)]

df_node = pd.DataFrame(index = df.index, columns = ct_labels + vol_labels)

print(df_node.shape)


# ## Load in data

# In[9]:


CT = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['CT_NAME_TMP'].replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(os.environ['CTDIR'], file_name))
    if i == 0: print(full_path)    

    ct = np.loadtxt(full_path[0])
    CT[i,:] = ct
    
df_node.loc[:,ct_labels] = CT


# In[11]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[12]:


VOL = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['VOL_NAME_TMP'].replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(os.environ['VOLDIR'], file_name))
    if i == 0: print(full_path)    

    img = nib.load(full_path[0])
    v = np.array(img.dataobj)
    v = v[v != 0]
    unique_elements, counts_elements = np.unique(v, return_counts=True)
    if len(unique_elements) == num_parcels:
        VOL[i,:] = counts_elements
    else:
        print(str(index) + '. Warning: not all parcels present')
        subj_filt[i] = True
    
df_node.loc[:,vol_labels] = VOL


# In[10]:


df_node.head()


# Save out brain feature data before any nuisance regression

# In[11]:


df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))


# ### Nuisance regression

# Regress out nuisance covariates. For cortical thickness, we regress out whole brain volume as well as average rating of scan quality from Roalf et al. 2018 NeuroImage.
# 
# We also used a mixed linear model for nuisance regression (instead of a simple OLS) to account for dependency in some of the data

# In[12]:


nuis = ['mprage_antsCT_vol_TBV', 'averageManualRating']
df_nuis = df.loc[:,nuis]
df_nuis = sm.add_constant(df_nuis)

for i, col in enumerate(df_node.columns):
    update_progress(i/df_node.shape[1])
    mdl = sm.MixedLM(df_node.loc[:,col], df_nuis, groups = df.reset_index()['bblid']).fit()
    y_pred = mdl.predict(df_nuis)
    df_node.loc[:,col] = df_node.loc[:,col] - y_pred
update_progress(1)


# In[13]:


df_node.head()


# ## Save out

# In[14]:


df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_clean.csv'))


# In[11]:


df_node = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_node_clean.csv'))


# In[13]:


df_node.set_index(['bblid','scanid'], inplace = True)


# In[15]:


df_node.head()


# In[24]:


from func import summarise_network

df_system = summarise_network(df_node, parcel_loc, yeo_idx, metrics = ('ct',), method = 'mean')

df_system = pd.concat((df_node.filter(regex = 'ct', axis = 1).mean(axis = 1).rename('ct'),df_system),axis = 1)


# In[25]:


df_system.to_csv('/Users/lindenmp/Dropbox/Work/git/python_cookbook/data/df_long_system.csv')


# In[26]:


df.to_csv('/Users/lindenmp/Dropbox/Work/git/python_cookbook/data/df_long_pheno.csv')

