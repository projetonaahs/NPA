import os
import numpy as np
import pandas as pd
from nilearn import datasets, maskers, connectome
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def generate_combined_matrix(func_directory, confounds_file, num_time_points):
    
    a = datasets.fetch_atlas_msdl()
    a_fname = a['maps']
    labels = a['labels']

    m4sk3r = maskers.NiftiMapsMasker(
        maps_img=a_fname, standardize='zscore_sample', verbose=5
    )

    
    all_t_series = []

    
    for func_file in os.listdir(func_directory):
        if func_file.endswith(".nii.gz"):
            
            func_path = os.path.join(func_directory, func_file)

            m4sk3r.fit(func_path)

            
            confounds_df = pd.read_csv(confounds_file)
            subject_id = func_file.split("_")[1]
            confounds_current = confounds_df[confounds_df['SUB_ID'] == int(subject_id)]

            if len(confounds_current) != num_time_points:
                print(f"Ajustando confounds para o sujeito {subject_id}...")
                if len(confounds_current) > num_time_points:
                    confounds_current = confounds_current.iloc[:num_time_points]
                else:
                    num_missing_rows = num_time_points - len(confounds_current)
                    missing_rows = pd.DataFrame([[0] * len(confounds_df.columns)] * num_missing_rows,
                                                columns=confounds_df.columns)
                    confounds_current = pd.concat([confounds_current, missing_rows], ignore_index=True)

            t_series = m4sk3r.transform(func_path, confounds=confounds_current)
            all_t_series.append(t_series)

    
    combined_t_series = np.concatenate(all_t_series, axis=0)

    
    c_measure = connectome.ConnectivityMeasure(kind='correlation')
    average_c_matrix = c_measure.fit_transform([combined_t_series])[0]
    np.fill_diagonal(average_c_matrix, 0)

    return average_c_matrix, labels


func_directory = '/home/julia/Documentos/ABIDE_pcp/cpac/nofilt_noglobal/control/sdsu'
confounds_file = '/home/julia/Documentos/ABIDE_pcp/cpac/Phenotypic_V1_0b_preprocessed1.csv'
num_time_points = 176


combined_c_matrix, labels = generate_combined_matrix(func_directory, confounds_file, num_time_points)


scaler = MinMaxScaler(feature_range=(0, 1))
combined_c_matrix = scaler.fit_transform(combined_c_matrix)


np.save('c_matrix_control.npy', combined_c_matrix)


fig_matrix, ax_matrix = plt.subplots(figsize=(12, 10))
cax_matrix = ax_matrix.matshow(combined_c_matrix, cmap='viridis')
fig_matrix.colorbar(cax_matrix, shrink=0.8, aspect=20)
ax_matrix.set_title("control connectivity matrix")

ax_matrix.set_xticks(np.arange(len(labels)))
ax_matrix.set_yticks(np.arange(len(labels)))
ax_matrix.set_xticklabels(labels, rotation=90)
ax_matrix.set_yticklabels(labels)

plt.show()
