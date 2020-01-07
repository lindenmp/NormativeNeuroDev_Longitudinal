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
from func import mark_outliers, get_cmap, run_corr, get_fdr_p, perc_dev, evd, summarise_network


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


# In[36]:


df_z_sys = summarise_network(df_z.loc[:,region_filter], parcel_loc[region_filter], yeo_idx[region_filter], metrics = ('ct',), method = 'mean')


# In[37]:


R = pd.DataFrame(index = df_z_sys.columns, columns = df_z_sys.columns)

for i_col in df_z_sys.columns:
    for j_col in df_z_sys.columns:
        R.loc[i_col,j_col] = sp.stats.pearsonr(df_z_sys[i_col],df_z_sys[j_col])[0]


# In[38]:


sns.heatmap(R.astype(float), annot = False, center = 0, vmax = 1, vmin = -1)


# # Predictive model: regression

# Here the goal is to use brain features (and demographics) from a given timepoint to predict each of our psychopathology phenotypes at a given timepoint.

# In[39]:


df.columns


# In[40]:


# First, we start by using T1 to predict T1
X = df_z_sys.loc[idx_t1,:].copy()
print(X.shape)

pheno = phenos[0]; print(pheno)
y = df.loc[idx_t1,pheno + '_n']

np.all(X.index.get_level_values(0) == y.index.get_level_values(0))


# In[41]:


# Add continous*continous interaction terms
X_int = X.multiply(df.loc[idx_t1,'scanageMonths'], axis = 0)
X_int_names = list(X.columns)
X_int_names = [s + '_age' for s in X_int_names]
X_int.columns = X_int_names


# In[42]:


# X = X_int.copy()
# X = pd.concat((X,X_int), axis = 1)


# In[43]:


nan_filt = y.isna().values
X = X[~nan_filt]; print(X.shape)
y = y[~nan_filt]; print(y.shape)

sc = StandardScaler()
X_std = sc.fit_transform(X); print(X_std.shape)


# In[44]:


# How does the Ridge regression alpha param impact coefficients within training sample?
n_alphas = 50
alphas = np.logspace(-5, 5, n_alphas)
# alphas = 10**np.linspace(10,-2,100)*0.5
print(alphas[0],alphas[-1])
coefs = []
for a in alphas:
    mdl = Ridge(alpha=a)
    mdl.fit(X_std, y)
    coefs.append(mdl.coef_)


# In[45]:


# inner_cv = KFold(n_splits=10, shuffle=True, random_state=0)
# outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)
inner_cv = KFold(n_splits=5, shuffle=False)
outer_cv = KFold(n_splits=5, shuffle=False)


# In[46]:


# scoring = {'r2': 'r2', 'mse': make_scorer(mean_squared_error), 'mae': make_scorer(mean_absolute_error)}
scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'}


# In[47]:


alpha_candidates = dict(alpha = alphas)
mdl = GridSearchCV(estimator=Ridge(), param_grid=alpha_candidates, cv=inner_cv, scoring = scoring, refit = 'r2')


# In[48]:


# Fit the cross validated grid search on the data 
mdl.fit(X_std, y);


# In[49]:


sns.set(style='white', context = 'talk', font_scale = 0.8)
f, ax = plt.subplots(2,2)
f.set_figwidth(15)
f.set_figheight(15)

ax[0,0].set_title('Ridge coefficients as a function of the regularization')
ax[0,0].plot(alphas, coefs)
ax[0,0].axvline(mdl.best_estimator_.alpha, linestyle = ':', color = 'k')
ax[0,0].set_xscale('log')
ax[0,0].set_xlabel('alpha')
ax[0,0].set_ylabel('weights')
ax[0,0].legend(list(X.columns))

ax[0,1].plot(alphas, mdl.cv_results_['mean_test_r2'])
ax[0,1].axvline(mdl.best_estimator_.alpha, linestyle = ':', color = 'k')
ax[0,1].set_xscale('log')
ax[0,1].set_xlabel('alpha')
ax[0,1].set_ylabel('r2')

ax[1,0].plot(alphas, mdl.cv_results_['mean_test_mse'])
ax[1,0].axvline(mdl.best_estimator_.alpha, linestyle = ':', color = 'k')
ax[1,0].set_xscale('log')
ax[1,0].set_xlabel('alpha')
ax[1,0].set_ylabel('neg mse')

ax[1,1].plot(alphas, mdl.cv_results_['mean_test_mae'])
ax[1,1].axvline(mdl.best_estimator_.alpha, linestyle = ':', color = 'k')
ax[1,1].set_xscale('log')
ax[1,1].set_xlabel('alpha')
ax[1,1].set_ylabel('neg mae')


# In[ ]:




