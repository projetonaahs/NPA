import pandas as pd


num_time_points = 196 

confounds_file = '/home/julia/Documentos/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
confounds_df = pd.read_csv(confounds_file)

rows_before = len(confounds_df)

if rows_before != num_time_points:
    if rows_before > num_time_points:
        confounds_df = confounds_df.iloc[:num_time_points]
    else:
        num_missing_rows = num_time_points - rows_before
        missing_rows = pd.DataFrame([[0] * len(confounds_df.columns)] * num_missing_rows, columns=confounds_df.columns)
        confounds_df = pd.concat([confounds_df, missing_rows], ignore_index=True)

rows_after = len(confounds_df)

print(f"número de linhas no arquivo de confounds antes do ajuste: {rows_before}")
print(f"número de linhas no arquivo de confounds após o ajuste: {rows_after}")


