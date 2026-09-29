# Dependencias y atribuciones

Este repositorio usa dependencias externas instaladas mediante sus distribuidores.
No incluye sus entornos virtuales ni una copia del código fuente de nnU-Net.

| Recurso | Fuente principal |
|---|---|
| PyTorch | https://pytorch.org/ |
| MONAI, UNet y SegResNet | https://github.com/Project-MONAI/MONAI |
| nnU-Net | https://github.com/MIC-DKFZ/nnUNet |
| BraTS 2023 | https://www.synapse.org/Synapse:syn51156910/wiki/622351 |
| Streamlit | https://github.com/streamlit/streamlit |
| NumPy, SciPy, pandas | https://numpy.org/ ; https://scipy.org/ ; https://pandas.pydata.org/ |
| Matplotlib, Plotly | https://matplotlib.org/ ; https://plotly.com/python/ |
| NiBabel, scikit-image, scikit-learn | https://nipy.org/nibabel/ ; https://scikit-image.org/ ; https://scikit-learn.org/ |

Los datos BraTS deben obtenerse por el canal autorizado y usarse conforme a sus
condiciones. Las imágenes MRI no están incluidas. Los términos del dataset y las
dependencias no se sustituyen por LICENSE.md.

HD-BET es una integración opcional del prototipo y no interviene en la evaluación
cuantitativa: https://github.com/MIC-DKFZ/HD-BET . No se distribuyen sus pesos.

Al citar los experimentos, cite también las publicaciones originales de los métodos
y del dataset empleadas en la bibliografía de la tesis. CITATION.cff corresponde
al acompañamiento computacional de esta tesis, no reemplaza esas referencias.
