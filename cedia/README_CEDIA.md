# Ejecución desde terminal / CEDIA

Los tres modelos y sus cinco folds ya fueron entrenados y evaluados. Este documento reproduce el protocolo final. Consulte [resultados](../docs/RESULTADOS.md) y [reproducibilidad](../docs/REPRODUCIBILIDAD.md).

## Configuración y cohorte

Desde la raíz del repositorio, edite la ubicación del dataset autorizado:

```bash
cp cedia/config.example.sh cedia/config.local.sh
nano cedia/config.local.sh
source cedia/config.local.sh
mkdir -p "$RUNS" logs
python3 scripts/prepare_cohort.py --source "$DATA_ROOT_ORIGINAL" --manifest "$MANIFEST" --output "$DATA_ROOT"
```

El helper selecciona exclusivamente los casos del manifiesto e implementa el alias `-segs.nii.gz` → `-seg.nii.gz` mediante enlaces. Use `--copy` si el sistema no admite enlaces; consume espacio adicional. Verifica tamaños registrados y no modifica las fuentes. No genere otras particiones ni importe el CSV histórico para reproducir estos resultados.

## Entorno GPU

Si Singularity ya proporciona el entorno usado, consérvelo y verifique versiones. Para un entorno nuevo compatible con CUDA 12.1:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r cedia/requirements.txt
python -m pip check
python -m cedia.cli preflight --data-root "$DATA_ROOT" --manifest "$MANIFEST" --deep --require-cuda --output "$RUNS/preflight_gpu.json"
python -m pip freeze > "$RUNS/environment_freeze.txt"
```

Solicite GPU, CPU, memoria y tiempo con los parámetros reales del servicio. No entrene en el nodo de acceso. Dentro de Singularity puede no existir `sbatch` aunque ya tenga una asignación interactiva. Los tamaños del nodo no equivalen a los recursos asignados.

## U-Net y SegResNet: entrenamiento nuevo

Perfil publicado `baseline`: 20 épocas, batch 4, 4 workers, Adam 1e-4, sin scheduler ni early stopping. Este bloque entrena y evalúa secuencialmente los diez folds y se detiene ante un error:

```bash
(
set -euo pipefail
for ARCH in unet segresnet; do
  for FOLD in 1 2 3 4 5; do
    export ARCH FOLD
    PROFILE=baseline bash cedia/run_fold.sh 2>&1 | tee -a "logs/${ARCH}_fold_${FOLD}.log"
  done
done
)
```

Para continuar, reactive el mismo entorno, cargue `source cedia/config.local.sh` y repita el bloque. Una época interrumpida se repite desde el último checkpoint. Conserve `last.pth`. No cambie batch, ROI, épocas o versiones dentro del mismo directorio. El perfil opcional `extended` no forma parte de los resultados publicados.

## Evaluar pesos publicados sin entrenar

Primero instale los paquetes como indica [PESOS.md](../docs/PESOS.md), incluidos sus metadatos:

```bash
(
set -euo pipefail
for ARCH in unet segresnet; do
  for FOLD in 1 2 3 4 5; do
    NUM=$(printf '%02d' "$FOLD")
    python -m cedia.cli evaluate --data-root "$DATA_ROOT" --manifest "$MANIFEST" --fold "$FOLD" --arch "$ARCH" --name "${ARCH}_baseline" --checkpoint "$RUNS/train/${ARCH}_baseline/fold_${NUM}/best.pth" --output "$RUNS/eval/${ARCH}_baseline" --save-predictions --resume
  done
done
)
```

Cada estudio se evalúa con su propio fold; no utilice un ensemble de los cinco outer folds para evaluar la misma cohorte.

## nnU-Net 2.6.2

Use un entorno separado y restricciones registradas:

```bash
python3.10 -m venv .venv_nnunet
source .venv_nnunet/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-nnunet.txt -c provenance/environment/constraints_nnunet.txt
python -m pip check
source cedia/config.local.sh
export VENV_DIR="$PROJECT_DIR/.venv_nnunet"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
python -m cedia.cli nnunet-export --data-root "$DATA_ROOT" --manifest "$MANIFEST" --output "$RUNS/nnunet" --link-mode symlink
```

Para entrenamiento nuevo, preparar cinco datasets independientes. El outer holdout no participa en el planner:

```bash
(
set -euo pipefail
for FOLD in 1 2 3 4 5; do
  ID=$((700+FOLD))
  DATASET="Dataset${ID}_GliomaOuter${FOLD}"
  nnUNetv2_plan_and_preprocess -d "$ID" -c 3d_fullres --verify_dataset_integrity -np 4
  python -m cedia.cli nnunet-install-split --dataset-dir "$nnUNet_raw/$DATASET" --preprocessed-dir "$nnUNet_preprocessed/$DATASET"
done
for FOLD in 1 2 3 4 5; do
  export FOLD
  TRAINER=nnUNetTrainer_100epochs bash cedia/run_nnunet_fold.sh
done
)
```

Cada dataset usa únicamente el fold interno `0` de su split personalizado. Compare los planes nuevos con los archivados en `results/nnunet/nnUNet_preprocessed` antes de atribuirlos al mismo protocolo. Los folds 3 y 4 usaron parches 128×160×112; 1, 2 y 5, 128³. No se garantiza identidad del planner con otro hardware/entorno. El perfil publicado tiene 100 épocas, no las 1.000 del trainer estándar.

Para inferir con pesos publicados: instálelos, ejecute `nnunet-export` y use este bloque sin planificar ni entrenar:

```bash
(
set -euo pipefail
for FOLD in 1 2 3 4 5; do
  DATASET="Dataset$((700+FOLD))_GliomaOuter${FOLD}"
  python -m cedia.cli nnunet-predict --data-root "$DATA_ROOT" --manifest "$MANIFEST" --fold "$FOLD" --dataset-dir "$nnUNet_raw/$DATASET" --trained-model "$nnUNet_results/$DATASET/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres" --output "$RUNS/nnunet_predictions" --resume
  python -m cedia.cli nnunet-evaluate --data-root "$DATA_ROOT" --manifest "$MANIFEST" --fold "$FOLD" --prediction-root "$RUNS/nnunet_predictions" --output "$RUNS/eval/nnunet" --resume
done
)
```

`checkpoint_best.pth` se selecciona con validación interna. Reanudar entrenamiento requiere `checkpoint_latest.pth` y puede repetir varias épocas. Los activos publicados solo incluyen `best`.

## Análisis final

```bash
python -m cedia.cli analyze --manifest "$MANIFEST" --inputs "$RUNS/eval/unet_baseline" "$RUNS/eval/segresnet_baseline" "$RUNS/eval/nnunet" --output "$RUNS/comparacion_tres_modelos" --bootstrap 2000
```

Exige cohorte, pacientes, folds y manifiesto compatibles. La evaluación de pesos originales, disponible en el CLI por trazabilidad, no forma parte de esta comparación.

## Ejecución por lotes

`slurm_job.sh` activa `VENV_DIR` y ejecuta el comando recibido. Complete `SLURM_OPTIONS` con recursos válidos de su cuenta. En el host con `sbatch`, después de exportar configuración:

```bash
export ARCH=unet PROFILE=baseline
sbatch "${SLURM_OPTIONS[@]}" --array=1-5%1 --output=logs/unet-%A_%a.out cedia/slurm_job.sh bash cedia/run_fold.sh
```

El entorno y los datos deben ser accesibles desde el nodo. El lanzamiento de Singularity y sus bind mounts dependen del servicio. Una sesión interactiva está sujeta a desconexión y límite de tiempo; `tee` no prolonga la asignación.
