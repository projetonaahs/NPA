import numpy as np
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.plotting import plot_anat, plot_epi, plot_glass_brain

# Função para o pré-processamento do arquivo de fMRI
def preprocess_fmri(file_path, activation_map):
    # Carregar a imagem fMRI
    fmri_img = image.load_img(file_path)

    # Aplicar técnicas de pré-processamento (exemplo: remoção de ruído)
    # Substitua esta seção com as técnicas de pré-processamento desejadas

    # Exibir a imagem anatômica
    plot_anat(fmri_img, title='Imagem Anatômica')

    # Exibir a imagem funcional pré-processada
    plot_epi(fmri_img, title='Imagem fMRI Pré-processada')

    # Exibir um mapa de ativação 3D com sobreposição
    glass_brain = plot_glass_brain(fmri_img, title='Mapa de Ativação')
    glass_brain.add_overlay(activation_map, cmap='hot', alpha=0.7)  # Adiciona a sobreposição com cor mais forte

    # Exibir os resultados
    plt.show()

# Caminho do arquivo de fMRI
file_path = './data/sub-004_T1w.nii.gz'

# Caminho do arquivo do mapa de ativação
activation_map_path = './data/sub-004_run-1_fieldmap.nii.gz'

# Carregar o mapa de ativação
activation_map = image.load_img(activation_map_path)

# Chamar a função de pré-processamento com a sobreposição de cor
preprocess_fmri(file_path, activation_map)