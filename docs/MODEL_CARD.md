# Model card: segmentación binaria de glioma

## Propósito y ámbito

El experimento apoya la tesis *Development of Customized Three-Dimensional Models of Gliomas for Preoperative Anatomical Visualization*, de **Yonny Josue Mera Macias**, Yachay Tech University. Evalúa segmentación whole-tumor, volumetría de máscaras y visualización anatómica exploratoria. Los modelos son artefactos de investigación; no se ha demostrado seguridad o utilidad clínica.

El repositorio contiene dos componentes distintos:

1. **Benchmark final por paciente:** U-Net, SegResNet y nnU-Net entrenados de nuevo en cinco folds, con checkpoints propios y resultados en `results/`.
2. **Prototipo histórico:** interfaz Streamlit y reconstrucción de superficies, vinculadas a los checkpoints originales. Los checkpoints del benchmark no sustituyen automáticamente a los de la aplicación. Las métricas finales no validan el funcionamiento clínico del prototipo.

## Entrada y salida

Cuatro volúmenes MRI alineados, en orden **T1 nativo, T1 con contraste, T2, FLAIR** (`t1n`, `t1c`, `t2w`, `t2f`). La referencia whole-tumor se obtiene como `label > 0`; la salida es una máscara binaria, de la cual se obtiene volumen en mL. La evaluación usa una cuadrícula común RAS, 1 mm isotrópico, recortada según el foreground de imagen y con padding para dimensiones divisibles por 16.

La máscara puede incluir edema y otras subregiones de la anotación. No estima márgenes microscópicos, resecabilidad, tractos, áreas funcionales, arterias o senos venosos. La superficie 3D del prototipo depende de la máscara y de operaciones posteriores de malla.

## Datos y particiones

1.250 estudios de 1.132 pacientes; cinco folds externos; aproximadamente 15% de los pacientes de cada pool externo de entrenamiento reservados para validación interna. Todas las adquisiciones de un paciente permanecen en el mismo rol. Cada estudio recibe una única predicción externa; no se evaluó un ensemble de los cinco folds sobre esta cohorte.

El archivo [`provenance/manifest_por_paciente.json`](../provenance/manifest_por_paciente.json) conserva las asignaciones exactas. Su huella lógica `manifest_sha256` es:

```text
f67351ece0c762564abc2fda59ffa8c2c05483e5a11fecd2a545f947348fc0da
```

La etiqueta `reconstructed_from_current_inventory_unverified_for_original_checkpoints` se refiere a los pesos históricos, entrenados antes de corregir las particiones. Los modelos nuevos se entrenaron con este manifiesto y sus registros conservan la misma huella. No debe editarse el manifiesto para eliminar esa etiqueta.

## Configuraciones entrenadas

| Propiedad | U-Net | SegResNet | nnU-Net |
|---|---|---|---|
| Implementación | MONAI 1.5.2 `UNet` | MONAI 1.5.2 `SegResNet` | nnU-Net v2 2.6.2, `3d_fullres` |
| Canales iniciales / features | `(16,32,64,128,256)` | 16; bloques down `(1,2,2,4)`, up `(1,1,1)` | `PlainConvUNet`, seis etapas `(32,64,128,256,320,320)` |
| Normalización de red | Batch norm | Group norm, 8 grupos | Instance norm |
| Épocas por fold | 20 | 20 | 100 |
| Batch de loader | 4 estudios, 2 patches por estudio | 4 estudios, 2 patches por estudio | 2 patches |
| Patch | `128 × 128 × 128` | `128 × 128 × 128` | Depende del fold; tabla inferior |
| Optimizador | Adam, LR `1e-4` | Adam, LR `1e-4` | SGD, LR inicial `0.01`, momentum `0.99`, Nesterov, weight decay `3e-5` |
| Scheduler / parada temprana | Ninguno / desactivada | Ninguno / desactivada | Scheduler de nnU-Net; trainer de 100 épocas |
| Selección de checkpoint | Mejor Dice medio de estudios en validación interna de volumen completo | Mismo criterio | Mejor EMA de pseudo-Dice de foreground |
| Predicción | Sigmoid `>0.5` | Sigmoid `>0.5` | Máscara binaria, `checkpoint_best.pth`, sin mirroring/TTA |

U-Net utiliza cuatro strides de 2 y dos unidades residuales por etapa. U-Net y SegResNet comparten `DiceCELoss` con pesos Dice/CE `0.7/0.3`, sigmoid, predicciones al cuadrado en el denominador Dice y smoothing `1e-5`. La normalización de imagen es z-score por canal sobre vóxeles no nulos; interpolación lineal de imágenes y nearest-neighbor de etiquetas. El muestreo usa pesos iguales para centros positivos/negativos, flips por eje, rotaciones de 90° y ruido gaussiano. Un batch completo de cuatro estudios produce ocho patches; eso no implica que la mitad de los vóxeles sean tumor.

nnU-Net utiliza sus transformaciones, pérdida Dice/CE y deep supervision, con 250 iteraciones de entrenamiento y 50 de validación por época. La planificación de cada dataset usa solo entrenamiento y validación interna; los estudios externos quedan fuera de la planificación. El split personalizado corresponde al **fold interno 0**, diferente del número de fold externo.

| Fold externo | Dataset nnU-Net | Patch `3d_fullres` | Stride de la última etapa |
|---:|---|---|---|
| 1 | `Dataset701_GliomaOuter1` | `[128,128,128]` | `[2,2,2]` |
| 2 | `Dataset702_GliomaOuter2` | `[128,128,128]` | `[2,2,2]` |
| 3 | `Dataset703_GliomaOuter3` | `[128,160,112]` | `[2,2,1]` |
| 4 | `Dataset704_GliomaOuter4` | `[128,160,112]` | `[2,2,1]` |
| 5 | `Dataset705_GliomaOuter5` | `[128,128,128]` | `[2,2,2]` |

Todos los planes guardados utilizan spacing `[1,1,1]`, batch 2 y `ZScoreNormalization` por canal. Los archivos `plans.json`, `cedia_protocol.json`, `dataset.json` y los registros de entrenamiento están conservados en `results/nnunet/`. Estos planes por fold precisan la descripción general de `128³` incluida en la tesis anterior a esta exportación; véase la [aclaración documental](RESULTADOS.md#precisión-documental-confirmada-al-recibir-la-exportación-de-cedia).

## Evaluación y reproducibilidad

El entorno registrado en CEDIA fue Linux/Singularity, Python 3.10.20, PyTorch 2.5.1+cu121, CUDA runtime 12.1, MONAI 1.5.2, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, NiBabel 5.4.2 y NVIDIA A100-SXM4-40GB. nnU-Net utilizó un entorno virtual separado. Conservar versiones, semillas y checkpoints permite auditar el protocolo, pero no garantiza identidad bit a bit entre hardware o ejecuciones distintas.

MONAI usa sliding-window `128³`, solapamiento 50% y blending constante. La máscara exportada por nnU-Net se remuestrea con nearest-neighbor a la cuadrícula común de evaluación. Surface Dice usa tolerancia exploratoria de 1 mm; HD95 es el máximo de los percentiles dirigidos entre bordes de centros de vóxel, y ASSD agrupa las distancias dirigidas. Los casos no finitos se contabilizan explícitamente.

Los [resultados](RESULTADOS.md) contienen medias por paciente, intervalos y limitaciones. Los quince pesos seleccionados se distribuyen fuera del árbol Git como archivos de release, junto con sus metadatos y hashes. Su función es evaluar los modelos seleccionados. Un checkpoint `best` no equivale necesariamente al último estado de entrenamiento necesario para reanudar exactamente una ejecución interrumpida.

## Usos no evaluados y limitaciones

No se validaron datos externos, segmentación multiclase, incertidumbre calibrada, imágenes incompletas o desalineadas, revisión de expertos, planificación de resección, navegación intraoperatoria ni beneficio quirúrgico. Las diferencias entre nnU-Net y las otras dos redes incluyen presupuesto de entrenamiento y decisiones de procesamiento; no son un estudio controlado del efecto aislado de la arquitectura. Los pesos históricos del prototipo no tienen la independencia por paciente del benchmark final.

Los derechos sobre datos, código de terceros y modelos deben considerarse por separado. Esta ficha describe procedencia y comportamiento observado; no asigna una licencia adicional a esos materiales.
