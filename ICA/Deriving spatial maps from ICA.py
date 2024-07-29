from nilearn import datasets
import os
from nilearn.image import load_img
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_prob_atlas
from nilearn import plotting


data=load_img('/Users/oj/Desktop/post_fMRI_RBD/sub-01/ses-1/func/sub-01_ses-1_task-BRAINMRINONCONTRASTDIFFUSION_acq-AxialfMRIrest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii')


canica = CanICA(
    n_components=10,
    memory="nilearn_cache",
    memory_level=2,
    verbose=10,
    mask_strategy="whole-brain-template",
    random_state=0,
    standardize="zscore_sample",
    n_jobs=2,
)
canica.fit(data)

canica_components_img = canica.components_img_


# Plot all ICA components together

from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

for i, cur_img in enumerate(iter_img(canica_components_img)):
    plot_stat_map(
        cur_img,
        display_mode="z",
        title=f"IC {int(i)}",
        cut_coords=1,
        colorbar=False,
    )
    plotting.show()

