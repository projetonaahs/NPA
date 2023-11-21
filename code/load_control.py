from nilearn import datasets

data = datasets.fetch_abide_pcp(data_dir='#', DX_GROUP=2, n_subjects=100, global_signal_regression=True, HANDEDNESS_CATEGORY='L') #substituir parametro data_dir por pasta (local) desejada para armazenar os arquivos do grupo de controle