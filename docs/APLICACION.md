# Prototipo de visualización

La aplicación conserva el backend histórico de U-Net y los avisos de investigación
de la tesis. Las métricas finales de los tres modelos corresponden al CLI de
CEDIA; no se atribuyen automáticamente al ensemble de la interfaz.

## Iniciar

En el entorno científico descrito en [REPRODUCIBILIDAD.md](REPRODUCIBILIDAD.md):

```bash
python -m pip install -r requirements.txt
python scripts/install_weights.py --archive ../05_pesos_prototipo_historico.tar.gz --output .
python -m streamlit run app.py
```

Introduzca una carpeta local con cuatro NIfTI del mismo estudio, alineados, con
sufijos `-t1n.nii.gz`, `-t1c.nii.gz`, `-t2w.nii.gz` y `-t2f.nii.gz`. Los datos no se
descargan desde esta aplicación. La inferencia puede usar CUDA si está disponible;
CPU será más lenta y el volumen completo requiere memoria suficiente.

HD-BET es opcional para extracción cerebral y debe instalarse por separado si se
activa. No es parte de la comparación cuantitativa. La reconstrucción cerebral
suavizada ofrece contexto visual; no representa corteza precisa, tractos,
función o vasos. WT incluye edema y no delimita tejido resecable.

No se distribuye un `.exe` nuevo ni se afirma validación clínica o prueba integral
de la interfaz sobre MRI en el equipo local. Los scripts de empaquetado Windows
antiguos se omiten para evitar confundirlos con una aplicación recompilada.
