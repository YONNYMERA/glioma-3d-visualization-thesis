# Checkpoints y activos de publicación

Los pesos se distribuyen como archivos adjuntos de una Release, separados del Git
del código. `provenance/weights.json` registra nombres, tamaños y SHA-256. No se
inventa una URL: los activos estarán disponibles cuando el autor publique la Release.

| Archivo | Contenido |
|---|---|
| `02_pesos_unet_baseline.tar.gz` | Cinco `best.pth` de U-Net, protocolo final |
| `03_pesos_segresnet_baseline.tar.gz` | Cinco `best.pth` de SegResNet, protocolo final |
| `04_pesos_nnunet.tar.gz` | Cinco `checkpoint_best.pth`, datasets 701–705, fold interno 0 |
| `05_pesos_prototipo_historico.tar.gz` | Cinco pesos originales para la aplicación histórica |

Estos paquetes públicos se volvieron a empaquetar sin el inventario privado de
CEDIA. Sus hashes de archivo difieren de la exportación privada, pero cada
checkpoint conserva exactamente sus bytes y su hash.

## Instalación del benchmark

Descargue los tres activos junto al repositorio o indique su ruta real. Desde la
raíz, con Python 3.10 y sin GPU:

```bash
python scripts/install_weights.py --archive ../02_pesos_unet_baseline.tar.gz
python scripts/install_weights.py --archive ../03_pesos_segresnet_baseline.tar.gz
python scripts/install_weights.py --archive ../04_pesos_nnunet.tar.gz
```

El destino predeterminado es `results_reproduction`. Si `RUNS` apunta a otra
carpeta, agregue `--output "$RUNS"`. El instalador verifica el tar, cada checkpoint,
rutas y archivos existentes antes de escribir; incluye los metadatos adyacentes
de `results`. No sobrescribe archivos distintos ni deserializa PyTorch.

Necesita espacio temporal adicional para verificar el contenido. En CEDIA puede
establecer `TMPDIR` en una carpeta de trabajo con espacio suficiente. Se copia un
paquete por invocación; repetirlo comprueba y omite archivos idénticos.

Para inferencia siga [cedia/README_CEDIA.md](../cedia/README_CEDIA.md). Estos
checkpoints seleccionados no constituyen un respaldo completo para continuar
entrenamientos: no se distribuyen `last.pth` ni `checkpoint_latest.pth`.

## Prototipo histórico

```bash
python scripts/install_weights.py --archive ../05_pesos_prototipo_historico.tar.gz --output .
```

Coloca cinco `best_metric_model_fold_*.pth` en `MODELS`. No son los pesos usados
para las métricas finales agrupadas por paciente. La aplicación usa un ensemble
U-Net histórico; no admite directamente SegResNet o nnU-Net.

La recepción de pesos permite comprobar su identidad con los registros. No
sustituye la verificación de inferencia sobre NIfTI ni supone validación clínica.
