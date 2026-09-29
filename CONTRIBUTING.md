# Contribuciones

Describa el problema, el comportamiento esperado y una reproducción mínima con
datos sintéticos. No adjunte imágenes médicas ni credenciales a issues.

Antes de proponer cambios:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s cedia/tests -v
```

Mantenga los resultados publicados inmutables y escriba nuevas ejecuciones en
otra carpeta. Un cambio de cohorte, transformación, modelo, pérdida o selección
de checkpoint constituye otro experimento y debe documentarse como tal.

La licencia de reutilización está pendiente de elección del autor; consulte
LICENSE.md antes de proponer una distribución derivada.
