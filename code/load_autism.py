from nilearn import datasets

data = datasets.fetch_abide_pcp(data_dir='/home/julia/Documentos/autism', DX_GROUP=1, n_subjects=100, global_signal_regression=True, HANDEDNESS_CATEGORY='L')