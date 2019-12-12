#import nispat
import os
import sys
import numpy as np
sys.path.append('/scratch/kg98/Linden/ResProjects/NormativeNeuroDev_Longitudinal/code/nispat/nispat')
from normative import estimate
from normative_parallel import execute_nm, collect_nm, rerun_nm_m3, delete_nm

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
wdir = normativedir; os.chdir(wdir)
collect_nm(wdir, collect=True)

delete_nm(wdir)
