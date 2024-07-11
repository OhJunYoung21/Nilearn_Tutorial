import nilearn
from nilearn import datasets
import os
from nilearn import image
import pandas as pd

atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]

# Loading the functional datasets
img = image.load_img('/Users/oj/Desktop/post_fMRI_RBD/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii')

confounds = pd.read_csv('/Users/oj/Desktop/post_fMRI_RBD/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_desc-confounds_timeseries.tsv',sep='\t')

confounds = confounds.fillna(0,inplace=True)

from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

time_series = masker.fit_transform(img, confounds=confounds)

#Compute the sparse inverase covariance

try:
    from sklearn.covariance import GraphicalLassoCV
except ImportError:
    # for Scitkit-Learn < v0.20.0
    from sklearn.covariance import GraphLassoCV as GraphicalLassoCV

estimator = GraphicalLassoCV()
estimator.fit(time_series)

from nilearn import plotting

# Display the covariance
'''
# The covariance can be found at estimator.covariance_
plotting.plot_matrix(
    estimator.covariance_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Sparse Inverse Covariance",
)
plotting.show()




plotting.plot_connectome(estimator.covariance_, coords, title="Covariance")

plotting.show()

plotting.plot_matrix(
    -estimator.precision_,
    labels=labels,
    figure=(9, 7),
    vmax=1,
    vmin=-1,
    title="Sparse inverse covariance",
)

plotting.show()
'''

coords = atlas.region_coords

plotting.plot_connectome(
    -estimator.precision_, coords, title="Sparse inverse covariance"
)

plotting.show()

