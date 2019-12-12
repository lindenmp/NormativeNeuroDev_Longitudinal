#!/usr/bin/env python
# coding: utf-8

# # Results, section 2:

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_Longitudinal/code/func/')
from proj_environment import set_proj_env
from func import mark_outliers, get_cmap, run_corr, get_fdr_p, perc_dev, evd


# In[3]:


exclude_str = 't1Exclude'
parc_str = 'schaefer'
parc_scale = 400
primary_covariate = 'scanageYears'
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str,
                                                                            parc_str = parc_str, parc_scale = parc_scale,
                                                                            primary_covariate = primary_covariate)


# In[4]:


os.environ['NORMATIVEDIR']


# In[5]:


metrics = ('ct',)
phenos = ('Overall_Psychopathology','Mania','Depression','Psychosis_Positive','Psychosis_NegativeDisorg')


# ## Setup plots

# In[6]:


if not os.path.exists(os.environ['FIGDIR']): os.makedirs(os.environ['FIGDIR'])
os.chdir(os.environ['FIGDIR'])
sns.set(style='white', context = 'talk', font_scale = 1)
cmap = sns.color_palette("pastel", 3)


# ## Load data

# In[7]:


# Train
df_train = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'train.csv'))
df_train.set_index(['bblid', 'scanid', 'timepoint'], inplace = True); print(df_train.shape)
df_node_train = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'resp_train.csv'))
df_node_train.set_index(['bblid', 'scanid', 'timepoint'], inplace = True); print(df_node_train.shape)

# Test
df = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'test.csv'))
df.set_index(['bblid', 'scanid', 'timepoint'], inplace = True); print(df.shape)
df_node = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'resp_test.csv'))
df_node.set_index(['bblid', 'scanid', 'timepoint'], inplace = True); print(df_node.shape)

# Normative model
smse = np.loadtxt(os.path.join(os.environ['NORMATIVEDIR'], 'smse.txt'), delimiter = ' ').transpose()
df_smse = pd.DataFrame(data = smse, index = df_node.columns)
z = np.loadtxt(os.path.join(os.environ['NORMATIVEDIR'], 'Z.txt'), delimiter = ' ').transpose()
df_z = pd.DataFrame(data = z, index = df_node.index, columns = df_node.columns)


# In[8]:


df.head()


# # Characterizing the psychopathology phenotype data

# Let's have a look at our psychopathology phenotype data, which are the continous DVs for our predictive model

# In[9]:


print('N:', df.shape[0])


# In[10]:


print('N w/ >=2 timepoints:', int(df.loc[df['TotalNtimepoints_new'] == 2,:].shape[0]/2 + df.loc[df['TotalNtimepoints_new'] == 3,:].shape[0]/3))
print('N w/ 3 timepoints:', int(df.loc[df['TotalNtimepoints_new'] == 3,:].shape[0]/3))


# In[11]:


# How much missing data have I got in the phenotypes?
for pheno in phenos:
    print('No. of NaNs for ' + pheno + ':', df.loc[:,pheno].isna().sum())


# In[12]:


my_bool = df.loc[:,phenos].isna().all(axis = 1)


# In[13]:


# For now I'm going to just drop the NaN rows. Need to look into data imputation methods for dependent data
df = df.loc[~my_bool,:]


# In[14]:


print('N:', df.shape[0])


# Need recalculate the num timepoints now...

# In[15]:


keep_me = ([1,2],[1,2,3])
idx_keep = []
idx_drop = []
for idx, data in df.groupby('bblid'):
    my_list = list(data.index.get_level_values(2).values)
    if my_list == keep_me[0] or my_list == keep_me[1]:
        idx_keep.append(idx)
    else:
        idx_drop.append(idx)


# In[16]:


df = df.loc[idx_keep,:]


# In[17]:


print('N:', df.shape[0])


# In[18]:


for idx, data in df.groupby('bblid'):
    df.loc[idx,'TotalNtimepoints_new'] = int(data.shape[0])
df.loc[:,'TotalNtimepoints_new'] = df.loc[:,'TotalNtimepoints_new'].astype(int)


# In[19]:


print('N w/ >=2 timepoints:', int(df.loc[df['TotalNtimepoints_new'] == 2,:].shape[0]/2 + df.loc[df['TotalNtimepoints_new'] == 3,:].shape[0]/3))
print('N w/ 3 timepoints:', int(df.loc[df['TotalNtimepoints_new'] == 3,:].shape[0]/3))


# Hasn't hurt my numbers too badly.

# In[20]:


df_node = df_node.loc[df.index,:]
df_z = df_z.loc[df.index,:]


# Now let's look at the distributions

# In[21]:


# Generate booleans that subset the data by timepoints
idx_t1 = df.index.get_level_values(2) == 1
idx_t2 = df.index.get_level_values(2) == 2
idx_t3 = df.index.get_level_values(2) == 3


# In[22]:


f, axes = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

for i, pheno in enumerate(phenos):
    sns.distplot(df.loc[idx_t1,pheno], ax = axes[i], color = cmap[0])
    sns.distplot(df.loc[idx_t2,pheno], ax = axes[i], color = cmap[1])
    sns.distplot(df.loc[idx_t3,pheno], ax = axes[i], color = cmap[2])
    axes[i].set_xlabel(pheno)
    axes[i].legend(['t1','t2','t3'])


# Clear issues of normality.

# In[23]:


for pheno in phenos:
    df.loc[:,pheno + '_n'] = sp.stats.yeojohnson(df.loc[:,pheno])[0]
#     df.loc[:,pheno + '_n'] = np.log(df.loc[:,pheno] + (df.loc[:,pheno].abs().max()+1))
#     df.loc[:,pheno + '_n'] = (df.loc[:,pheno] - df.loc[:,pheno].mean())/df.loc[:,pheno].std()  


# In[24]:


f, axes = plt.subplots(1,len(phenos))
f.set_figwidth(len(phenos)*5)
f.set_figheight(5)

for i, pheno in enumerate(phenos):
    sns.distplot(df.loc[idx_t1,pheno + '_n'], ax = axes[i], color = cmap[0])
    sns.distplot(df.loc[idx_t2,pheno + '_n'], ax = axes[i], color = cmap[1])
    sns.distplot(df.loc[idx_t3,pheno + '_n'], ax = axes[i], color = cmap[2])
    axes[i].set_xlabel(pheno)
    axes[i].legend(['t1','t2','t3'])


# In[25]:


for i, pheno in enumerate(phenos):
#     print(np.var(df.loc[idx_t1,pheno + '_n']))
    x = df.loc[:,pheno + '_n']
    my_med = np.median(x)
    mad = np.median(abs(x - my_med))/1.4826
    print(mad)


# # Feature selection

# Starting by applying a set of reasonable baseline feature selection criteria:
# 
# 1) Regions where there is a significant relationship between age and the regional brain features in the training set
# 
# 2) Regions where the normative model was able to perform out of sample predictions (as index by standardized mean squared error < 1)
# 
# 3) Regions where extreme deviations occur

# ### 1) Age effects

# In[26]:


# age effect on training set
df_age_effect = run_corr(df_train[primary_covariate], df_node_train, typ = 'pearsonr'); df_age_effect['p_fdr'] = get_fdr_p(df_age_effect['p'])
if parc_str == 'lausanne':
    df_age_effect.drop(my_list, axis = 0, inplace = True)
age_alpha = 0.05
age_filter = df_age_effect['p_fdr'].values < age_alpha
age_filter.sum()


# ### 2) Normative model performance

# In[27]:


smse_thresh = 1
smse_filter = df_smse.values < smse_thresh
smse_filter = smse_filter.reshape(-1)
smse_filter.sum()


# ### 3) Extreme deviations

# At either time point 1 or 2

# In[28]:


ed_filter_t1 = perc_dev(df_z.loc[idx_t1,:].transpose()) > 1
print(ed_filter_t1.sum())
ed_filter_t2 = perc_dev(df_z.loc[idx_t2,:].transpose()) > 1
print(ed_filter_t2.sum())


# Combine all the filters into one

# In[29]:


region_filter_1 = np.logical_and(age_filter,smse_filter)
region_filter_1.sum()


# In[30]:


region_filter = np.logical_and(region_filter_1,np.logical_or(ed_filter_t1,ed_filter_t2))
region_filter.sum()


# # Feature summaries

# Alternatively could be to collapse over brain regions into single summary measures. There are a few obvious ways to do this: mean, extreme value stats. Let's look at a few!

# In[31]:


df_node_summary = pd.DataFrame(index = df_node.index)
for metric in metrics:
    df_node_summary[metric+'_node_mean'] = df_node.loc[:,region_filter].filter(regex = metric, axis = 1).mean(axis = 1)
    df_node_summary[metric+'_z_mean'] = df_z.loc[:,region_filter].filter(regex = metric, axis = 1).mean(axis = 1)
    df_node_summary[metric+'_z_evd'] = evd(df_node.loc[:,region_filter].filter(regex = metric, axis = 1))
    df_node_summary[metric+'_z_evd_pos'] = evd(df_node.loc[:,region_filter].filter(regex = metric, axis = 1), sign = 'pos')
    df_node_summary[metric+'_z_evd_neg'] = np.abs(evd(df_node.loc[:,region_filter].filter(regex = metric, axis = 1), sign = 'neg'))


# In[32]:


df_node_summary.head()


# How do the summaries relate?

# In[33]:


R = pd.DataFrame(index = df_node_summary.columns, columns = df_node_summary.columns)

for i_col in df_node_summary.columns:
    for j_col in df_node_summary.columns:
        R.loc[i_col,j_col] = sp.stats.pearsonr(df_node_summary[i_col],df_node_summary[j_col])[0]


# In[34]:


sns.heatmap(R.astype(float), annot = True, center = 0, vmax = 1, vmin = -1)


# In[35]:


g = sns.pairplot(df_node_summary, kind = 'reg', diag_kind = 'kde', height = 2)

