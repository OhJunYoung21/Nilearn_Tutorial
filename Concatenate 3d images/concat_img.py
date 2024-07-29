import os
from nilearn.image import concat_imgs
from nilearn import image
from nilearn import plotting

def concat_image(directory):
    files = [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(".nii.gz")]

    images = [image.load_img(f) for f in files]

    imgs = concat_imgs(images)

    return imgs

result = concat_image('/Users/oj/Desktop/pre_BIDS/BIDS_RBD/sub-37/func')

result.to_filename('/Users/oj/Desktop/pre_BIDS/BIDS_RBD/sub-37/func/concatenated_image.nii')