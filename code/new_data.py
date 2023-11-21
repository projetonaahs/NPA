import os
import numpy as np
import pandas as pd
from nilearn import datasets, input_data, connectome
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def generate_subject_matrix(func_directory, confounds_file):
    a = datasets.fetch_atlas_msdl()
    a_fname = a['maps']
    labels = a['labels']

    m4sk3r = input_data.NiftiMapsMasker(
        maps_img=a_fname, standardize='zscore_sample', verbose=5
    )

    func_files = [f for f in os.listdir(
        func_directory) if f.endswith(".nii.gz")]

    selected_subject = np.random.choice(func_files, 1)[0]
    func_path = os.path.join(func_directory, selected_subject)

    m4sk3r.fit(func_path)

    confounds_df = pd.read_csv(confounds_file)
    subject_id = selected_subject.split("_")[1]

    t_series = m4sk3r.transform(func_path)

    c_measure = connectome.ConnectivityMeasure(kind='correlation')
    c_matrix = c_measure.fit_transform([t_series])[0]
    np.fill_diagonal(c_matrix, 0)

    return c_matrix, labels


func_directory = '/home/julia/Documentos/control/ABIDE_pcp/cpac/nofilt_global'  
confounds_file = '/home/julia/Documentos/NPA/Phenotypic_V1_0b_preprocessed1.csv'

subject_c_matrix, labels = generate_subject_matrix(
    func_directory, confounds_file)

scaler = MinMaxScaler(feature_range=(0, 1))
subject_c_matrix = scaler.fit_transform(subject_c_matrix)

np.save('c_matrix_unknown.npy', subject_c_matrix)

fig_matrix, ax_matrix = plt.subplots(figsize=(12, 10))
cax_matrix = ax_matrix.matshow(subject_c_matrix, cmap='viridis')
fig_matrix.colorbar(cax_matrix, shrink=0.8, aspect=20)
ax_matrix.set_title("Unknown Connectivity Matrix")

ax_matrix.set_xticks(np.arange(len(labels)))
ax_matrix.set_yticks(np.arange(len(labels)))
ax_matrix.set_xticklabels(labels, rotation=90)
ax_matrix.set_yticklabels(labels)

plt.show()
