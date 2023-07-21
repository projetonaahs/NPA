from nilearn import datasets, plotting, maskers, connectome
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = datasets.fetch_adhd(n_subjects=7)

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

#calcula a matriz de conectividade funcional.
c_measure = connectome.ConnectivityMeasure(kind='correlation')
c_matrix = c_measure.fit_transform([t_series])[0]

#preenche de branco a diagonal da matriz e plota. - evitar equívocos.
np.fill_diagonal(c_matrix, 0)
plotting.plot_matrix(
    c_matrix, labels=labels, colorbar=True, vmax=0.8, vmin=-0.8
)

#plota o gráfico de connectome - apenas 15%.
coords = a.region_coords
plotting.plot_connectome(
    c_matrix, coords, edge_threshold='85%'
)

#dataframe com rótulos e regiões. *******arrumar*******
region_table = pd.DataFrame({'Rótulo': a['labels'], 'Função': a['description']})

html_table = region_table.to_html()

#salvar arquivo html.
with open('region_table.html', 'w') as file:
    file.write(html_table)

plotting.show()
