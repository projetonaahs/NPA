import numpy as np
import matplotlib.pyplot as plt
from nilearn import image, connectome
from nilearn.plotting import plot_anat, plot_epi, plot_matrix

# Função para o pré-processamento do arquivo de fMRI
def preprocess_fmri(file_path):
    # Carregar a imagem fMRI
    fmri_img = image.load_img(file_path)

    # Converter os dados de 3D para 2D
    fmri_data = fmri_img.get_fdata().reshape(-1, fmri_img.shape[-1])

    # Calcular a matriz de conectividade funcional
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([fmri_data])[0]

    # Exibir a imagem anatômica
    plot_anat(fmri_img, title='Imagem Anatômica')

    # Exibir a imagem funcional pré-processada
    plot_epi(fmri_img, title='Imagem fMRI Pré-processada')

    # Exibir a matriz de conectividade
    plot_matrix(correlation_matrix, title='Matriz de Conectividade')

    # Exibir os resultados
    plt.show()

# Caminho do arquivo de fMRI
file_path = '../../../../julizzz/Downloads/sub-736_ses-2_task-FOV_run-1_bold.nii.gz'

# Chamar a função de pré-processamento
preprocess_fmri(file_path)
