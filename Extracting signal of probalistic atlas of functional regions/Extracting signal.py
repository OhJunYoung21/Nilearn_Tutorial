from nilearn import datasets
import os
from nilearn import image
from nilearn.maskers import NiftiMapsMasker
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from nilearn import plotting

atlas = datasets.fetch_atlas_msdl()

atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# Load the functional datasets
img = image.load_img('/Users/oj/Desktop/post_fMRI_RBD/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii')

confounds = pd.read_csv('/Users/oj/Desktop/post_fMRI_RBD/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv',sep='\t')

confounds = confounds.fillna(0,inplace=True)


masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)
masker.fit(img)
time_series = masker.transform(img, confounds=confounds)

print(time_series.shape)

# 해당 결과가 (100,39)가 나왔으므로, 뇌를 39구역으로 나눴으며, 100번의 스캔이 이뤄졌음을 알 수 있다.

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize=True,
)
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)

coords = atlas.region_coords


plotting.plot_connectome(
    correlation_matrix, coords, edge_threshold="80%", colorbar=True
)

plotting.show()