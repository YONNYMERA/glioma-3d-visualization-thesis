# Datos y particiones

## Procedencia y acceso

La cohorte procede del recurso de entrenamiento **BraTS 2023 GLI**, identificado en la tesis como **Synapse `syn51156910`**. El usuario debe obtener las imágenes por el canal oficial y cumplir sus condiciones de acceso, uso y atribución. El repositorio no contiene los volúmenes MRI, las segmentaciones de referencia ni el dataset preprocesado. Los resultados tabulares y el manifiesto conservan identificadores de investigación BraTS, no nombres de pacientes.

La referencia bibliográfica del dataset y sus requisitos de cita deben tomarse de la versión obtenida oficialmente. Esta documentación no transfiere derechos sobre BraTS ni presenta las imágenes como datos propios del autor.

## Inventario final

La cohorte final contiene **1.250 estudios de 1.132 pacientes**. Cada estudio aporta cuatro modalidades (`t1n`, `t1c`, `t2w`, `t2f`) y una máscara de referencia. La tarea es binaria whole-tumor: todos los valores positivos de la referencia pertenecen al foreground.

Ejemplo de estructura descrita por el manifiesto:

```text
DATA_ROOT/
└── BraTS-GLI-00002-000/
    ├── BraTS-GLI-00002-000-t1n.nii.gz
    ├── BraTS-GLI-00002-000-t1c.nii.gz
    ├── BraTS-GLI-00002-000-t2w.nii.gz
    ├── BraTS-GLI-00002-000-t2f.nii.gz
    └── BraTS-GLI-00002-000-seg.nii.gz
```

La raíz es configurable, y las rutas del manifiesto son relativas a ella. Algunos archivos del material original tenían el sufijo `-segs.nii.gz`; el inventario final persistido referencia `-seg.nii.gz`. Al preparar los datos, debe respetarse la correspondencia de estudios, modalidades y referencias indicada en el manifiesto. No se debe modificar ni sustituir una máscara para resolver una discrepancia de nombre.

El orden de canales es T1 nativo, T1 con contraste, T2 y FLAIR. No se deben intercambiar `t2w` y `t2f`. El caso identifica una adquisición; en `BraTS-GLI-00002-000`, el prefijo `BraTS-GLI-00002` identifica al paciente y el último bloque a la adquisición.

## Particiones exactas por paciente

| Fold | Entrenamiento, estudios (pacientes) | Validación interna, estudios (pacientes) | Evaluación externa, estudios (pacientes) |
|---:|---:|---:|---:|
| 1 | 857 (769) | 147 (136) | 246 (227) |
| 2 | 846 (769) | 145 (136) | 259 (227) |
| 3 | 848 (770) | 150 (136) | 252 (226) |
| 4 | 848 (770) | 151 (136) | 251 (226) |
| 5 | 855 (770) | 153 (136) | 242 (226) |

Se aplicó KFold con cinco folds, barajado y semilla 0 a la lista ordenada de identificadores únicos de paciente. Dentro del pool de cada fold externo, aproximadamente 15% de los pacientes se reservaron para validación interna; la semilla interna fue `0 + número_de_fold`. Todas las adquisiciones siguen el rol de su paciente. Cada estudio aparece exactamente una vez en evaluación externa.

Para reproducir los resultados publicados, use [`manifest_por_paciente.json`](../provenance/manifest_por_paciente.json), sin reconstruir ni editar sus asignaciones. El campo `manifest_sha256` contiene:

```text
f67351ece0c762564abc2fda59ffa8c2c05483e5a11fecd2a545f947348fc0da
```

Es una huella del contenido lógico del manifiesto, calculada por el código del proyecto excluyendo el propio campo de huella. **No es el SHA256 de los bytes del archivo JSON ni un hash de los NIfTI.** El manifiesto registra rutas y tamaños de los archivos; no proporciona por sí solo hashes completos del contenido de las imágenes.

## Por qué existe un protocolo nuevo

La revisión del CSV original detectó adquisiciones diferentes de un mismo paciente en roles distintos dentro de un fold. No bastaba con evitar duplicados de identificadores de estudio. El inventario recuperado también contenía 1.250 estudios, frente a los 1.251 mencionados en la versión inicial de la tesis. Se corrigieron las particiones y se entrenaron de nuevo los tres modelos.

El campo de procedencia `reconstructed_from_current_inventory_unverified_for_original_checkpoints` se conserva literalmente: advierte que esas particiones no justifican la evaluación independiente de los checkpoints históricos. Los quince checkpoints del experimento final sí se entrenaron con el manifiesto corregido, registrado en sus metadatos.

No se debe atribuir a los pesos históricos la evidencia del experimento nuevo, añadir silenciosamente directorios de otra descarga a la cohorte ni regenerar una partición con un inventario diferente y presentarla como el mismo experimento.

## Verificaciones y alcance

La ejecución de CEDIA registró el preflight de geometría entre modalidades y referencia, valores finitos y etiquetas admitidas. Las asignaciones se validan para excluir solapamiento de pacientes entre roles y exigir una única evaluación externa por estudio.

La presente distribución permite revisar particiones, registros, métricas y procedencia de checkpoints. Para repetir predicciones y comprobar las métricas vóxel a vóxel se necesitan además las imágenes autorizadas y los pesos correspondientes. No se distribuye un conjunto institucional externo y no se deduce generalización clínica a partir de este inventario.
