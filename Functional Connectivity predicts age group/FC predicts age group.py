from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
from nilearn.connectome import ConnectivityMeasure



adhd_data = datasets.fetch_adhd(n_subjects=10)

msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords

masker = NiftiMapsMasker(
    msdl_data.maps,
    resampling_target="data",
    t_r=2,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    memory="nilearn_cache",
    memory_level=1,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
).fit()

masker.fit(adhd_data.func)

connectivity_data=[] # actiavtion patterns for all subjects
labels=[]  # 1 if ADHD, 0 if control

for func_file, confound_file, phenotypic in zip(
        adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)

    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    connectivity_data.append(correlation_matrix)
    labels.append(phenotypic['adhd'])

X_train,X_test,y_train,y_test = train_test_split(connectivity_data,labels,test_size=0.2,random_state=1)



