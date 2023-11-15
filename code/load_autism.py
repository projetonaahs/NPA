from nilearn import datasets

data = datasets.fetch_abide_pcp(data_dir='/home/julia/Documentos/ABIDE_pcp/cpac/nofilt_noglobal/autism', DX_GROUP=1, n_subjects=100)