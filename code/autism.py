from nilearn import datasets, maskers, connectome
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import glob



func_directory = '/home/julia/Documentos/ABIDE_pcp/cpac/nofilt_noglobal/autism/'

confounds_file = '/home/julia/Documentos/ABIDE_pcp/cpac/Phenotypic_V1_0b_preprocessed1.csv'

confounds_df = pd.read_csv(confounds_file)

num_time_points = 196

func_files_autism = glob.glob(func_directory + '*.nii.gz')[:9]

a = datasets.fetch_atlas_msdl()
a_fname = a['maps']
labels = a['labels']

m4sk3r = maskers.NiftiMapsMasker(
    maps_img=a_fname, standardize='zscore_sample', verbose=5 #set a memory
)
m4sk3r.fit(func_files_autism)  

connectivity_matrices = []

for func_file in func_files_autism:
    subject_id = func_file.split("/")[-1].split("_")[1]

    confounds_current = confounds_df[confounds_df['SUB_ID'] == int(subject_id)]

    if len(confounds_current) != num_time_points:
        print(f"ajustando confounds para o sujeito {subject_id}...")
        if len(confounds_current) > num_time_points:
            confounds_current = confounds_current.iloc[:num_time_points]
        else:
            num_missing_rows = num_time_points - len(confounds_current)
            missing_rows = pd.DataFrame([[0] * len(confounds_df.columns)] * num_missing_rows, columns=confounds_df.columns)
            confounds_current = pd.concat([confounds_current, missing_rows], ignore_index=True)

    t_series_autism = m4sk3r.transform(func_file, confounds=confounds_current)

    c_measure = connectome.ConnectivityMeasure(kind='correlation')
    c_matrix_autism = c_measure.fit_transform([t_series_autism])[0]

    np.fill_diagonal(c_matrix_autism, 0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    c_matrix_autism = scaler.fit_transform(c_matrix_autism)
    np.save('c_matrix_autism.npy', c_matrix_autism)


    connectivity_matrices.append(c_matrix_autism)

average_connectivity_matrix = np.mean(connectivity_matrices, axis=0)

fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.matshow(average_connectivity_matrix, cmap='viridis')
fig.colorbar(cax, shrink=0.8, aspect=20)
ax.set_title("autism correlation matrix")

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

threshold = 0.7
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        label1 = labels[i]
        label2 = labels[j]
        strength = average_connectivity_matrix[i, j]
        if strength >= threshold:
            print(f"{label1} + {label2}: {strength:.2f}")


#plt.show()
