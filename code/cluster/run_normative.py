#import nispat
import os
import sys
import numpy as np
sys.path.append('/scratch/kg98/Linden/ResProjects/NormativeNeuroDev_Longitudinal/code/nispat/nispat')
from normative import estimate
from normative_parallel import execute_nm, collect_nm, rerun_nm_m3, delete_nm

# ------------------------------------------------------------------------------
# parallel (batch)
# ------------------------------------------------------------------------------
# settings and paths
python_path = '/home/lindenmp/virtual_env/NeuroDev_NetworkControl/bin/python'
normative_path = '/scratch/kg98/Linden/ResProjects/NormativeNeuroDev_Longitudinal/code/nispat/nispat/normative.py'
batch_size = 2
memory = '4G'
duration = '1:00:00'
cluster_spec = 'm3'

# ------------------------------------------------------------------------------
# Normative dir
# ------------------------------------------------------------------------------
# primary directory for normative model
exclude_str = 't1Exclude'
combo_label = 'schaefer_400'

normativedir = os.path.join('/scratch/kg98/Linden/ResProjects/NormativeNeuroDev_Longitudinal/analysis/normative',
	exclude_str, combo_label, 'scanageYears+sex_adj/')

print(normativedir)

# ------------------------------------------------------------------------------
# Primary model
job_name = 'prim_'
wdir = normativedir; os.chdir(wdir)

# input files and paths
cov_train = os.path.join(normativedir, 'cov_train.txt')
resp_train = os.path.join(normativedir, 'resp_train.txt')
cov_test = os.path.join(normativedir, 'cov_test.txt')
resp_test = os.path.join(normativedir, 'resp_test.txt')

# run normative
execute_nm(wdir, python_path=python_path, normative_path=normative_path, job_name=job_name, covfile_path=cov_train, respfile_path=resp_train,
           batch_size=batch_size, memory=memory, duration=duration, cluster_spec=cluster_spec, cv_folds=None, testcovfile_path=cov_test, testrespfile_path=resp_test, alg = 'gpr')
