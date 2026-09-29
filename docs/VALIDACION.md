# Validación de la entrega para GitHub

Revisión local: 29 de septiembre de 2026. No se repitieron entrenamientos GPU.

| Comprobación | Resultado |
|---|---|
| Cuatro paquetes privados exportados de CEDIA | SHA-256 coincide con el inventario recibido |
| 15 pesos finales y cinco históricos | Bytes y SHA-256 verificados al crear los paquetes públicos |
| Identidad de los 15 pesos finales | Coincide con registros de entrenamiento y evaluación |
| Manifiesto | 1.250 estudios, 1.132 pacientes; sin pacientes compartidos entre roles |
| Evaluación externa | 15 folds completos; 3.750 filas caso-modelo, consistentes con tabla unificada |
| Pruebas del pipeline CEDIA | 27/27 aprobadas con datos sintéticos CPU |
| Pruebas de preparación/instalación | 15 aprobadas; una omitida por falta de privilegio de symlink de Windows |
| Instalación de paquetes reales | U-Net y nnU-Net instalados con hashes y metadatos verificados |
| Análisis bootstrap recalculado | Tablas de resumen, diferencias pareadas y acuerdo volumétrico coinciden con tolerancia 1e-11 |
| Generador portátil de figuras | Produce las siete figuras estadísticas; conteos y curvas parseadas coinciden |
| Sintaxis Python y enlaces Markdown | Verificados en el árbol distribuido |

Entorno local de verificación: Windows, Python 3.10.11, PyTorch 2.9.0+cpu,
MONAI 1.5.1. Es distinto del entorno de entrenamiento registrado en CEDIA
(Python 3.10.20, PyTorch 2.5.1+cu121, MONAI 1.5.2, A100). La receta de instalación
documenta las versiones de CEDIA; no se afirma haber reconstruido todo ese entorno
Linux en Windows. GitHub Actions está preparado para Python 3.10 y dependencias
CPU del protocolo, pero no se ha ejecutado en GitHub durante esta entrega local.

No se recibieron los NIfTI fuente ni se verificó aquí la inferencia completa
vóxel a vóxel. La instalación de pesos no deserializa checkpoints ni demuestra
por sí misma inferencia correcta. No se hizo prueba integral de la interfaz,
validación clínica, evaluación externa ni comparación justa de tiempos GPU.

`scripts/verify_repository.py --checksums` comprueba también el árbol exacto de
esta entrega mediante `SHA256SUMS.txt`. Los checksums deberán actualizarse al
publicar una revisión que cambie archivos; el control de cohortes sin esa opción
continúa siendo útil para las pruebas automáticas del repositorio.
