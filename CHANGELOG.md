# Registro de la entrega para GitHub

## 1.0.0 — 2026-09-29

- Integración del código ejecutado en CEDIA y registros de los 15 experimentos.
- Inclusión del manifiesto exacto, métricas, planes de nnU-Net, versiones y hashes.
- Guías actualizadas al protocolo final: 20 épocas MONAI y 100 nnU-Net.
- Helpers para reconstruir la cohorte y restaurar pesos verificados con sus metadatos.
- Figuras estadísticas reproducibles, model card, dataset card y citación.
- Separación entre comparación final y prototipo histórico; pesos en activos externos.
- Configuración portable, eliminación de rutas personales y salidas del notebook.
- Precisión metodológica: parches nnU-Net diferentes en folds 3 y 4 según planes recibidos.

No se modificaron las funciones numéricas del entrenamiento/evaluación de CEDIA
ni se repitió el entrenamiento GPU durante el empaquetado. La preparación para
GitHub no cambia las métricas ni implica validación clínica o externa.
