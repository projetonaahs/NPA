from nilearn import image

import pandas as pd


# data = '/home/julia/Documentos/npa/ABIDE_pcp/cpac/nofilt_noglobal/Pitt_0050005_func_preproc.nii.gz'
# func_img = image.load_img(data)
# num_time_points = func_img.shape[-1]

# print(f"Número de pontos no tempo nos dados funcionais: {num_time_points}")



# confounds_file = '/home/julia/Documentos/npa/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
# confounds_data = pd.read_csv(confounds_file)
# num_rows = confounds_data.shape[0]

# print(f"Número de linhas no arquivo CSV de confounds: {num_rows}")
import pandas as pd

# Carregue seus dados funcionais e obtenha o número de pontos no tempo
num_time_points = 196  # Substitua pelo número real de pontos no tempo

# Carregue o arquivo de confounds
confounds_file = '/home/julia/Documentos/npa/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
confounds_df = pd.read_csv(confounds_file)

# Verifique o número de linhas no arquivo de confounds antes do ajuste
num_confounds_rows_before = len(confounds_df)

# Ajuste o arquivo de confounds, se necessário
if num_confounds_rows_before != num_time_points:
    if num_confounds_rows_before > num_time_points:
        # Exclua linhas extras
        confounds_df = confounds_df.iloc[:num_time_points]
    else:
        # Preencha as linhas ausentes com zeros (ou outros valores apropriados)
        num_missing_rows = num_time_points - num_confounds_rows_before
        missing_rows = pd.DataFrame([[0] * len(confounds_df.columns)] * num_missing_rows, columns=confounds_df.columns)
        confounds_df = pd.concat([confounds_df, missing_rows], ignore_index=True)

# Verifique o número de linhas no arquivo de confounds após o ajuste
num_confounds_rows_after = len(confounds_df)

# Imprima o número de linhas antes e depois do ajuste
print(f"Número de linhas no arquivo de confounds antes do ajuste: {num_confounds_rows_before}")
print(f"Número de linhas no arquivo de confounds após o ajuste: {num_confounds_rows_after}")

# Agora confounds_df contém o arquivo de confounds ajustado
