import os
import numpy as np
import pandas as pd
from nilearn import datasets, input_data, connectome
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def generate_combined_matrix(func_directory, confounds_file, num_time_points, num_subjects):
    a = datasets.fetch_atlas_msdl()
    a_fname = a['maps']
    labels = a['labels']

    m4sk3r = input_data.NiftiMapsMasker(
        maps_img=a_fname, standardize='zscore_sample', verbose=5
    )

    all_t_series = []

    func_files = sorted([f for f in os.listdir(func_directory) if f.endswith(".nii.gz")])

    for func_file in func_files[:num_subjects]:
        func_path = os.path.join(func_directory, func_file)

        m4sk3r.fit(func_path)

        confounds_df = pd.read_csv(confounds_file)
        subject_id = func_file.split("_")[1]

        t_series = m4sk3r.transform(func_path)
        all_t_series.append(t_series)

    combined_t_series = np.concatenate(all_t_series, axis=0)

    c_measure = connectome.ConnectivityMeasure(kind='correlation')
    average_c_matrix = c_measure.fit_transform([combined_t_series])[0]
    np.fill_diagonal(average_c_matrix, 0)

    return average_c_matrix, labels

func_directory = '/home/julia/Documentos/autism/ABIDE_pcp/cpac/nofilt_global/'
confounds_file = '/home/julia/Documentos/NPA/Phenotypic_V1_0b_preprocessed1.csv'
num_time_points = 176
num_subjects = 10

combined_c_matrix, labels = generate_combined_matrix(func_directory, confounds_file, num_time_points, num_subjects)

# Normalização
scaler = MinMaxScaler(feature_range=(0, 1))
combined_c_matrix = scaler.fit_transform(combined_c_matrix)

# Salvando a matriz normalizada em um arquivo Numpy
np.save('c_matrix_autism.npy', combined_c_matrix)

# Visualização da matriz de conectividade
fig_matrix, ax_matrix = plt.subplots(figsize=(12, 10))
cax_matrix = ax_matrix.matshow(combined_c_matrix, cmap='viridis')
fig_matrix.colorbar(cax_matrix, shrink=0.8, aspect=20)
ax_matrix.set_title("Autism Connectivity Matrix")

ax_matrix.set_xticks(np.arange(len(labels)))
ax_matrix.set_yticks(np.arange(len(labels)))
ax_matrix.set_xticklabels(labels, rotation=90)
ax_matrix.set_yticklabels(labels)

# Imprime as conexões acima de 0.7
threshold = 0.7
for i, label1 in enumerate(labels):
    for j, label2 in enumerate(labels):
        if i < j and combined_c_matrix[i, j] > threshold:
            print(f"{label1} + {label2}: {combined_c_matrix[i, j]:.2f}")

plt.show()
