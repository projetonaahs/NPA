from nilearn import datasets

data = datasets.fetch_abide_pcp(data_dir='#', DX_GROUP=1, n_subjects=30, global_signal_regression=True, HANDEDNESS_CATEGORY='L') #replace the parameter data_dir with the desired folder (local) to store the files for the autism group.