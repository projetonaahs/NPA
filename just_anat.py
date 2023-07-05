import numpy as np
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.plotting import plot_anat, plot_epi, plot_glass_brain

# Função para o pré-processamento do arquivo de fMRI
def preprocess_fmri(file_path):
    # Carregar a imagem fMRI
    fmri_img = image.load_img(file_path)

    # Aplicar técnicas de pré-processamento (exemplo: remoção de ruído)
    # Substitua esta seção com as técnicas de pré-processamento desejadas

    # Exibir a imagem anatômica
    plot_anat(fmri_img, title='Imagem Anatômica')

    # Exibir a imagem funcional pré-processada
    plot_epi(fmri_img, title='Imagem fMRI Pré-processada')

    # Exibir um mapa de ativação 3D
    glass_brain = plot_glass_brain(fmri_img, title='Mapa de Ativação')
    glass_brain.add_overlay(fmri_img)

    # Exibir os resultados
    plt.show()

# Caminho do arquivo de fMRI
file_path = './data/3d/sub26_anat.nii.gz'

# Chamar a função de pré-processamento
preprocess_fmri(file_path)