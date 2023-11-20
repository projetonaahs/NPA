from nilearn import datasets

data = datasets.fetch_abide_pcp(data_dir='/home/julia/Documentos/control', DX_GROUP=2, n_subjects=100, global_signal_regression=True, HANDEDNESS_CATEGORY='L')