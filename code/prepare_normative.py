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
from func import get_synth_cov


# In[3]:


exclude_str = 't1Exclude'
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str)


# In[4]:


print(os.environ['MODELDIR'])
if not os.path.exists(os.environ['MODELDIR']): os.makedirs(os.environ['MODELDIR'])


# ## Load data

# In[5]:


df = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_pheno.csv'))
df.set_index(['bblid', 'scanid', 'timepoint'], inplace = True)

df_node = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))
df_node.set_index(['bblid', 'scanid', 'timepoint'], inplace = True)

# adjust sex to 0 and 1
df['sex_adj'] = df.sex - 1
print(df.shape)
print(df_node.shape)


# # Prepare files for normative modelling

# Use the train/test split defined in get_longitudinal_sample to split data files and prepare inputs from normative modelling.
# 
# Here, even though the data includes dependency (due to longitudinal data), the normative model is trained on cross-sectional data meaing the assumption of independence is stills satisfied for the gaussian process regression. However, the deviations from the normative model (which are estimated independently over individuals and regions) will have dependencies and this needs to be accounted for in subsequent analyses.

# In[6]:


# Note, 'ageAtScan1_Years' is assumed to be covs[0] and 'sex_adj' is assumed to be covs[1]
# if more than 2 covs are to be used, append to the end and age/sex will be duplicated accordingly in the forward model
covs = ['scanageYears', 'sex_adj']

print(covs)
num_covs = len(covs)
print(num_covs)


# In[7]:


extra_str_2 = ''


# ## Primary model (train/test split)

# In[8]:


# Create subdirectory for specific normative model -- labeled according to parcellation/resolution choices and covariates
normativedir = os.path.join(os.environ['MODELDIR'], '+'.join(covs) + extra_str_2 + '/')
print(normativedir)
if not os.path.exists(normativedir): os.mkdir(normativedir)


# In[9]:


# Write out training
df.loc[~df['train_test'],:].to_csv(os.path.join(normativedir, 'train.csv'))
df.loc[~df['train_test'],:].to_csv(os.path.join(normativedir, 'cov_train.txt'), columns = covs, sep = ' ', index = False, header = False)

# Write out test
df.loc[df['train_test'],:].to_csv(os.path.join(normativedir, 'test.csv'))
df.loc[df['train_test'],:].to_csv(os.path.join(normativedir, 'cov_test.txt'), columns = covs, sep = ' ', index = False, header = False)


# In[10]:


resp_train = df_node.loc[~df['train_test'],:]
mask = np.all(np.isnan(resp_train), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_train.to_csv(os.path.join(normativedir, 'resp_train.csv'))
resp_train.to_csv(os.path.join(normativedir, 'resp_train.txt'), sep = ' ', index = False, header = False)

resp_test = df_node.loc[df['train_test'],:]
mask = np.all(np.isnan(resp_test), axis = 1)
if np.any(mask): print("Warning: NaNs in response train")
resp_test.to_csv(os.path.join(normativedir, 'resp_test.csv'))
resp_test.to_csv(os.path.join(normativedir, 'resp_test.txt'), sep = ' ', index = False, header = False)

print(str(resp_train.shape[1]) + ' features written out for normative modeling')


# ### Forward variants

# Used only to examine the predictions made by the trained normative model

# In[11]:


# Create subdirectory for specific normative model -- labeled according to parcellation/resolution choices and covariates
fwddir = os.path.join(normativedir, 'forward/')
if not os.path.exists(fwddir): os.mkdir(fwddir)

# Synthetic cov data
x = get_synth_cov(df, cov = 'scanageYears', stp = 1)

if 'sex_adj' in covs:
    # Produce gender dummy variable for one repeat --> i.e., to account for two runs of ages, one per gender
    gender_synth = np.concatenate((np.ones(x.shape),np.zeros(x.shape)), axis = 0)

# concat
synth_cov = np.concatenate((np.matlib.repmat(x, 2, 1), np.matlib.repmat(gender_synth, 1, 1)), axis = 1)
print(synth_cov.shape)

# write out
np.savetxt(os.path.join(fwddir, 'synth_cov_test.txt'), synth_cov, delimiter = ' ', fmt = ['%.1f', '%.d'])

