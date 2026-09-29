# Reproducibilidad

## Comprobar la entrega sin dependencias científicas

Python 3.10, desde la raíz:

```bash
python scripts/verify_repository.py
python -m unittest discover -s tests -v
```

El manifiesto contiene 1.250 estudios de 1.132 pacientes y cinco particiones. Su SHA interno es `f67351ece0c762564abc2fda59ffa8c2c05483e5a11fecd2a545f947348fc0da`. Los identificadores pertenecen al dataset BraTS. No se distribuyen imágenes.

## Entorno científico CPU

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r cedia/requirements.txt
python -m pip check
python -m unittest discover -s cedia/tests -v
```

En Windows, activar con `.venv\Scripts\Activate.ps1`. Las pruebas usan datos sintéticos y no necesitan BraTS. [provenance/environment](../provenance/environment) contiene el entorno observado en CEDIA. Los `pip freeze` incluyen paquetes ajenos al experimento: son evidencia histórica, no una receta mínima portátil.

El experimento usó Python 3.10.20, PyTorch 2.5.1+cu121, MONAI 1.5.2 y A100 de 40 GB. Para GPU instale la rueda compatible antes de las demás dependencias; véase [PyTorch](https://pytorch.org/get-started/previous-versions/). No se garantiza igualdad bit a bit con otro hardware/entorno.

## Recalcular análisis y figuras incluidos

No requiere GPU, MRI ni pesos:

```bash
python -m cedia.cli analyze --manifest provenance/manifest_por_paciente.json --inputs results/eval/unet_baseline results/eval/segresnet_baseline results/eval/nnunet --output results_reproduction/comparacion_tres_modelos --bootstrap 2000
python scripts/generate_figures.py --results results --output generated_figures
```

El primer comando recalcula tablas, bootstrap y gráficos del análisis. El segundo genera las siete figuras estadísticas: curvas de ambos pipelines, comparación de métricas, distribuciones, matrices de confusión, concordancia volumétrica y estratos de tamaño. Los PDF/PNG publicados y leyendas están en [docs/figures](figures).

Las matrices agrupan voxels y se normalizan por clase de referencia; no son matrices diagnósticas por paciente. Las curvas de validación interna no son resultados del conjunto externo. Los tiempos de MONAI y nnU-Net tienen alcances distintos.

## Repetir inferencia o entrenamiento

Siga [cedia/README_CEDIA.md](../cedia/README_CEDIA.md) y [PESOS.md](PESOS.md). Use `results_reproduction` para no mezclar la evidencia entregada con nuevas ejecuciones. Los pesos `best` sirven para inferencia. Para reanudar un entrenamiento nuevo debe conservar sus `last.pth`/`checkpoint_latest.pth`, que no se distribuyen como activos de publicación.

## Procedencia

Código y resultados proceden de la exportación final de CEDIA del 29 de septiembre de 2026. Se comprobaron las huellas de los paquetes recibidos y los checkpoints seleccionados. `provenance/source_files.json` registra hashes de origen y copia pública; las rutas de usuario se sustituyeron por variables en metadatos y logs. El manifiesto se conserva byte a byte. Los checksums solo cubren esta entrega; modificaciones posteriores requieren actualizarlos.

nnU-Net se instala como dependencia, sin redistribuir su entorno. Se omiten 3.750 CSV individuales redundantes, las predicciones NIfTI y el dataset. Los 15 `metrics.csv` y la tabla unificada permiten rehacer el análisis.
