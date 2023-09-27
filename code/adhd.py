from nilearn import datasets, maskers, connectome
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = datasets.fetch_adhd(n_subjects=10)

func_file = data.func[0]

confounds_file = data.confounds[0]

a = datasets.fetch_atlas_msdl()
a_fname = a['maps']
labels = a['labels']

m4sk3r = maskers.NiftiMapsMasker(
    maps_img=a_fname, standardize='zscore_sample', memory='nilearn_cache', verbose=5
)
m4sk3r.fit(func_file)
t_series = m4sk3r.transform(func_file, confounds=confounds_file)

# Calcula a matriz de conectividade funcional.
c_measure = connectome.ConnectivityMeasure(kind='correlation')
c_matrix = c_measure.fit_transform([t_series])[0]

# Preenche de branco a diagonal da matriz
np.fill_diagonal(c_matrix, 0)

# Normaliza os valores da matriz de correlação para o intervalo de 0 a 1
scaler = MinMaxScaler(feature_range=(0, 1))
c_matrix = scaler.fit_transform(c_matrix)

# Configuração da figura para a matriz de correlação
fig_matrix, ax_matrix = plt.subplots(figsize=(8, 6))
cax_matrix = ax_matrix.matshow(c_matrix, cmap='coolwarm')
fig_matrix.colorbar(cax_matrix, shrink=0.8, aspect=20)
ax_matrix.set_title("Matriz de Correlação Normalizada (0-1)")

# Configuração dos rótulos das regiões
ax_matrix.set_xticks(np.arange(len(labels)))
ax_matrix.set_yticks(np.arange(len(labels)))
ax_matrix.set_xticklabels(labels, rotation=90)
ax_matrix.set_yticklabels(labels)

# Exibe as partes do cérebro envolvidas na análise

# Cria um gráfico vazio apenas para os rótulos e força
fig_legend, ax_legend = plt.subplots(figsize=(6, 6))
ax_legend.axis('off')

# Calcula e configura as forças das conexões
legend_text = "\n".join([f"{label1} + {label2}: {c_matrix[i, j]:.2f}" for i, label1 in enumerate(labels) for j, label2 in enumerate(labels) if i < j])
ax_legend.text(0.1, 0.1, legend_text, fontsize=10)

# Exibe a primeira janela com a matriz de correlação e a segunda janela com os rótulos e a força das conexões
plt.show()
