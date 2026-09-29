# Resultados del experimento final

La evaluación corresponde a segmentación binaria **whole tumor** en 1.250 estudios de resonancia de 1.132 pacientes de BraTS 2023 GLI. Cada estudio se evaluó una sola vez con el modelo de su fold externo. Las adquisiciones de un paciente permanecieron juntas en cada partición. Estos resultados sustituyen a los del experimento histórico, cuyas particiones no garantizaban independencia por paciente.

## Resultado principal

Las cifras siguientes son medias por paciente: primero se promedian las adquisiciones de cada paciente y después los pacientes. No son medias de vóxeles ni promedios simples de los cinco folds.

| Modelo | Épocas por fold | Dice, media [IC 95%] | HD95 finito, mm | ASSD finito, mm | Surface Dice a 1 mm | Error volumétrico absoluto, mL |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 20 | 0,9013 [0,8949; 0,9071] | 8,8857 | 1,6922 | 0,8241 | 8,3417 |
| SegResNet | 20 | 0,9167 [0,9116; 0,9213] | 7,4924 | 1,4936 | 0,8543 | 7,0852 |
| nnU-Net | 100 | 0,9285 [0,9241; 0,9326] | 6,4070 | 1,1526 | 0,8762 | 6,1659 |

Fuente de los valores y de los intervalos de todas las métricas: [`summary_ci_and_size.csv`](../results/comparacion_tres_modelos/summary_ci_and_size.csv), filas `Size_group=all`. Cada modelo aporta 1.250 estudios y 1.132 pacientes para Dice.

![Comparación de métricas por paciente](figures/metric_comparison.png)

U-Net produjo una predicción vacía en `BraTS-GLI-00753-001`, fold 5: Dice y Surface Dice son cero, HD95 y ASSD son infinitos y la precisión es indefinida. Las medias de distancias y precisión de U-Net excluyen ese estudio para la métrica correspondiente; quedan 1.249 estudios finitos de 1.132 pacientes, porque ese paciente tiene otra adquisición evaluable. El fallo permanece en la media de Dice. Por tanto, «HD95 finito» no resume por sí solo todos los fallos.

## Comparaciones pareadas

| Diferencia de Dice | Media | IC 95% |
|---|---:|---:|
| SegResNet − U-Net | 0,01538 | [0,01219; 0,01865] |
| nnU-Net − U-Net | 0,02723 | [0,02371; 0,03091] |
| nnU-Net − SegResNet | 0,01186 | [0,00975; 0,01416] |

Las diferencias se calculan en estudios emparejados y se promedian dentro de cada paciente. Fuente: [`paired_patient_differences.csv`](../results/comparacion_tres_modelos/paired_patient_differences.csv). Para distancias, los pares deben ser finitos en ambos modelos; no se obtiene la diferencia pareada restando sin más las medias marginales de la tabla anterior.

Los IC percentiles proceden de 2.000 réplicas bootstrap de pacientes dentro de los folds externos fijos, con semilla `20260923`. Son condicionales a los checkpoints entrenados: no incluyen variabilidad de reentrenamiento ni de repetir la selección del modelo. Dice es la métrica principal; los demás intervalos son exploratorios, sin ajuste por comparaciones múltiples. El protocolo se conserva en [`analysis_protocol.json`](../results/comparacion_tres_modelos/analysis_protocol.json).

## Figuras y registros

| Archivo | Interpretación |
|---|---|
| [Curvas U-Net y SegResNet](figures/training_unet_segresnet.png) | Pérdida e internal-validation Dice de volumen completo; 20 épocas, cinco folds por modelo. La validación interna selecciona checkpoints y no es la evaluación externa. |
| [Curvas nnU-Net](figures/training_nnunet.png) | 100 épocas; pérdidas y pseudo-Dice por patches. Los índices originales 0–99 se muestran como épocas completadas 1–100. La curva de mejor EMA refleja mejoras registradas, no una trayectoria EMA completa. |
| [Matrices de confusión](figures/confusion_matrices.png) | Conteos de vóxeles agrupados en 1.250 estudios; porcentajes normalizados por fila de referencia. No son matrices de diagnóstico por paciente. |
| [Distribuciones](figures/metric_distributions.png) | Dice por paciente y HD95 finito; el eje de HD95 es logarítmico. |
| [Acuerdo volumétrico](figures/volumetric_agreement.png) | Volúmenes promediados por paciente, identidad y Bland–Altman; correlación no equivale a acuerdo. |
| [Estratos por tamaño](figures/size_strata.png) | Umbrales exploratorios de volumen de referencia: <10 mL, 10–<50 mL y ≥50 mL. |

Las versiones PDF de estas figuras se encuentran en la misma carpeta. Los registros por fold están en [`results/train`](../results/train), [`results/eval`](../results/eval) y [`results/nnunet`](../results/nnunet). Los CSV preservan mayor precisión numérica que las tablas de lectura.

Los estratos contienen 24, 297 y 929 estudios, respectivamente; son 22, 278 y 850 pacientes dentro de cada estrato. Un paciente puede aportar adquisiciones a más de un estrato, por lo que esos conteos de pacientes no se suman para obtener la cohorte. En el estrato pequeño, el Dice medio fue 0,6367, 0,7253 y 0,7861 para U-Net, SegResNet y nnU-Net; el pequeño tamaño muestral requiere una interpretación cautelosa.

## Alcance y limitaciones

- Se comparan tres protocolos completos. nnU-Net utilizó 100 épocas, planificación y procesamiento propios; U-Net y SegResNet utilizaron 20 épocas. No se aisló el efecto de la arquitectura ni se igualó el presupuesto computacional. Veinte épocas no demuestran convergencia.
- La evaluación es interna a esta cohorte. No hubo prueba externa institucional, revisión por especialistas, evaluación de usabilidad, medición de precisión de mallas ni resultados quirúrgicos.
- La máscara binaria reúne las subregiones anotadas; no identifica por separado tumor realzante, núcleo o edema ni demuestra un límite histológico.
- Las mediciones de tiempo tienen alcances distintos: MONAI mide sliding-window después de la transferencia; nnU-Net incluye preprocesamiento, inferencia, remuestreo y exportación después de la lectura. No se presenta una clasificación de velocidad entre los tres métodos.
- Las matrices incluyen el fondo del volumen RAS a 1 mm, recortado por foreground y rellenado para divisibilidad. Especificidad y NPV elevadas no equivalen a segmentación perfecta ni a precisión diagnóstica por paciente.
- Las figuras estadísticas no requieren distribuir los NIfTI originales. No se han incluido paneles individuales de resonancia en esta carpeta pública de figuras.

## Precisión documental confirmada al recibir la exportación de CEDIA

La tesis entregada previamente describe patches de nnU-Net de `128³` de manera general. Los archivos `plans.json` y `debug.json` recibidos para preparar este repositorio permiten precisar que los folds externos **3 y 4 usaron `128 × 160 × 112`**, mientras que los folds **1, 2 y 5 usaron `128 × 128 × 128`**. El último stride también difiere: `[2,2,1]` frente a `[2,2,2]`. Esta aclaración del protocolo no modifica los checkpoints ni las métricas publicadas. La especificación por fold figura en [MODEL_CARD.md](MODEL_CARD.md); las fuentes son los planes guardados en `results/nnunet/nnUNet_results/Dataset70N_GliomaOuterN/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/plans.json`.
