# Functions for project: neurodev_long
# Linden Parkes, 2019
# lindenmp@seas.upenn.edu

import os, sys
import numpy as np

def set_proj_env(dataset = 'PNC', train_test_str = 'squeakycleanExclude', exclude_str = 't1Exclude',
    parc_str = 'schaefer', parc_scale = 200, parc_variant = 'orig', 
    primary_covariate = 'scanageYears', extra_str = ''):

    # Project root directory
    projdir = '/Users/lindenmp/Dropbox/Work/ResProjects/neurodev_long'; os.environ['PROJDIR'] = projdir
    
    # Derivatives for dataset --> root directory for the dataset under analysis
    derivsdir = os.path.join('/Volumes/ResProjects_2TB/ResData/PNC/'); os.environ['DERIVSDIR'] = derivsdir

    # Parcellation specifications
    # Names of parcels
    if parc_str == 'schaefer': parcel_names = np.genfromtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames.txt'), dtype='str')

    # vector describing whether rois belong to cortex (1) or subcortex (0)
    if parc_str == 'schaefer': parcel_loc = np.loadtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames_loc.txt'), dtype='int')
    
    if parc_variant == 'no_bs':
        drop_parcels = np.where(parcel_loc == 2)
        num_parcels = parcel_names.shape[0] - drop_parcels[0].shape[0]
    elif parc_variant == 'cortex_only':
        drop_parcels = np.where(parcel_loc != 1)
        num_parcels = parcel_names.shape[0] - drop_parcels[0].shape[0]
    else:
        drop_parcels = []
        num_parcels = parcel_names.shape[0]

    # Cortical thickness directory
    ctdir = os.path.join(derivsdir, 'processedData/antsCorticalThickness'); os.environ['CTDIR'] = ctdir

    if parc_str == 'schaefer':
        ct_name_tmp = 'bblid/*xscanid/ct_schaefer' + str(parc_scale) + '_17.txt'; os.environ['CT_NAME_TMP'] = ct_name_tmp
        voldir = os.path.join(derivsdir, 'processedData/gm_vol_masks_native'); os.environ['VOLDIR'] = voldir
        vol_name_tmp = 'bblid/*xscanid/Schaefer2018_' + str(parc_scale) + '_17Networks_native_gm.nii.gz'; os.environ['VOL_NAME_TMP'] = vol_name_tmp

    # Normative dir based on the train/test split --> specific combinations of parcellation/number of parcels/edge weight come off this directory
    # This is the first of the output directories for the project and is created by get_train_test.ipynb if it doesn't exist
    basedir = os.path.join(projdir, 'analysis/normative', exclude_str); os.environ['BASEDIR'] = basedir

    # Subdirector in normative dir (basedir) that designates combination of parcellation/number of parcels/edge weight
    modeldir_base = os.path.join(basedir, parc_str + '_' + str(num_parcels)); os.environ['MODELDIR_BASE'] = modeldir_base
    modeldir = os.path.join(basedir, parc_str + '_' + str(num_parcels) + extra_str); os.environ['MODELDIR'] = modeldir

    # normative dir
    model = primary_covariate + '+sex_adj'
    normativedir = os.path.join(modeldir, model); os.environ['NORMATIVEDIR'] = normativedir

    # Figure directory
    figdir = os.path.join(modeldir, 'figs'); os.environ['FIGDIR'] = figdir

    # Yeo labels
    if parc_str == 'lausanne':
        yeo_idx = np.loadtxt(os.path.join(projdir, 'figs_support/labels/yeo7netlabelsLaus' + str(parc_scale) + '.txt')).astype(int)
        yeo_labels = ('Visual', 'Somatomator', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal Control', 'Default Mode', 'Subcortical', 'Brainstem')
    elif parc_str == 'schaefer':
        yeo_idx = np.loadtxt(os.path.join(projdir, 'figs_support/labels/yeo17netlabelsSchaefer' + str(parc_scale) + '.txt')).astype(int)
        yeo_labels = ('Vis. A', 'Vis. B', 'Som. Mot. A', 'Som. Mot. B', 'Dors. Attn. A', 'Dors. Attn. B', 'Sal. Vent. Attn. A', 'Sal. Vent. Attn. B',
                    'Limbic A', 'Limbic B', 'Cont. A', 'Cont. B', 'Cont. C', 'Default A', 'Default B', 'Default C', 'Temp. Par.')

    return parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels
