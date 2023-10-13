import pandas as pd
from nilearn import datasets, maskers, connectome
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/home/julia/Documentos/"

# Parâmetro para filtrar apenas indivíduos saudáveis (grupo de controle)
abide = datasets.fetch_abide_pcp(data_dir, n_subjects=10, DX_GROUP=2)



# Carregar o atlas MSDL
a = datasets.fetch_atlas_msdl()
a_fname = a.maps

# Inicializar o masker
m4sk3r = maskers.NiftiMapsMasker(
    maps_img=a_fname, standardize='zscore_sample', memory='nilearn_cache', verbose=5
)

# Extrair séries temporais das regiões definidas pelo atlas
t_series = m4sk3r.transform(abide.func_preproc)

# Calcular a matriz de conectividade funcional (correlação)
c_measure = connectome.ConnectivityMeasure(kind='correlation')
c_matrix = c_measure.fit_transform(t_series)

# Preencher a diagonal com zeros
np.fill_diagonal(c_matrix, 0)

# Normalizar a matriz de conectividade
scaler = MinMaxScaler(feature_range=(0, 1))
c_matrix = scaler.fit_transform(c_matrix)

# Renderizar a matriz como um gráfico de calor
fig_matrix, ax_matrix = plt.subplots(figsize=(12, 10))
cax_matrix = ax_matrix.matshow(c_matrix, cmap='viridis')  # Você pode escolher outro mapa de cores se preferir
fig_matrix.colorbar(cax_matrix, shrink=0.8, aspect=20)
ax_matrix.set_title("Matriz de Conectividade Funcional Normalizada (0-1)")

plt.show()
