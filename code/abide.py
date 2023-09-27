# Importando as bibliotecas necessárias
from nilearn import plotting, maskers, connectome, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib  # Para manipular arquivos NIfTI

# Substitua o caminho abaixo pelo caminho para o seu arquivo funcional (imagem funcional)
func_file_path = '../nilearn_data/adhd'

# Substitua o caminho abaixo pelo caminho para o seu arquivo de confounds
confounds_file_path = 'caminho/para/seu/arquivo/confounds.txt'

# Carregando um atlas (mapas cerebrais) chamado MSDL
a = datasets.fetch_atlas_msdl()
a_fname = a['maps']  # Obtendo os mapas do atlas
labels = a['labels']  # Obtendo as etiquetas (nomes) das regiões do atlas

# Criando um objeto de máscara usando os mapas do atlas
m4sk3r = maskers.NiftiMapsMasker(
    maps_img=a_fname, standardize='zscore_sample', memory='nilearn_cache', verbose=5
)

# Carregando os dados funcionais diretamente do arquivo local
func_img = nib.load(func_file_path)
confounds = pd.read_csv(confounds_file_path, delimiter='\t')  # Seu arquivo de confounds pode ser um arquivo CSV

m4sk3r.fit(func_img)  # Ajustando a máscara aos dados funcionais
t_series = m4sk3r.transform(func_img, confounds=confounds.values)  # Aplicando a máscara aos dados funcionais

# Calculando a matriz de conectividade funcional usando correlação
c_measure = connectome.ConnectivityMeasure(kind='correlation')
c_matrix = c_measure.fit_transform([t_series])[0]  # Obtendo a matriz de conectividade

# Preenchendo a diagonal da matriz com zeros para evitar autocorrelação
np.fill_diagonal(c_matrix, 0)

# Plotando uma matriz de conectividade com rótulos e barra de cores
plotting.plot_matrix(
    c_matrix, labels=labels, colorbar=True, vmax=0.8, vmin=-0.8
)

# Plotando o gráfico de connectome, exibindo apenas as conexões acima do limiar de 85%
coords = a.region_coords  # Obtendo as coordenadas das regiões
view = plotting.view_connectome(
    c_matrix, coords
)

# Exibindo as visualizações
view.open_in_browser()
