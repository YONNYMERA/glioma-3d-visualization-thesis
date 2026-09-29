# Publicar este repositorio

## Código

1. Descomprima `3D_Glioma_Thesis_GitHub.zip`. La carpeta que debe abrir como
   repositorio es `3D_Glioma_Thesis`, donde está `README.md`.
2. Cree un repositorio vacío en su cuenta. Nombre sugerido:
   `glioma-3d-visualization-thesis`. No agregue otro README ni licencia automática.
3. Use GitHub Desktop para añadir esa carpeta local, crear el primer commit y
   publicar. También puede usar Git desde la carpeta:

```bash
git init -b main
git add .
git commit -m "Add thesis code, patient-grouped benchmark and reproducibility records"
git remote add origin https://github.com/TU_USUARIO/glioma-3d-visualization-thesis.git
git push -u origin main
```

Sustituya `TU_USUARIO` por su cuenta real. No suba el ZIP como único archivo del
repositorio: GitHub debe mostrar las carpetas y el README descomprimidos.

Descripción sugerida: `Patient-grouped BraTS glioma segmentation benchmark and
research prototype for 3D anatomical visualization. Thesis code and reproducibility records.`
Temas sugeridos: `glioma`, `brats`, `medical-imaging`, `segmentation`, `monai`,
`nnunet`, `reproducible-research`, `thesis`, `streamlit`.

## Pesos

Cree una Release, por ejemplo `v1.0.0`, y adjunte los cuatro `.tar.gz` y el
`SHA256SUMS.txt` de la carpeta externa `archivos_para_Releases`. No los copie
dentro del Git del código. Los usuarios los instalarán con `scripts/install_weights.py`.

GitHub bloquea archivos individuales de más de 100 MiB en un repositorio normal;
la carga web de archivos del repositorio admite hasta 25 MiB. Las Releases son
el mecanismo elegido aquí para los binarios grandes. Fuente:
[documentación de GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Metadatos de publicación

El repositorio no necesita una URL inventada para funcionar localmente. Después
de publicarlo, puede agregar la URL real y el identificador de versión en
`CITATION.cff`. Un DOI solo debe añadirse cuando exista un depósito que lo emita.

No se ha escogido una licencia de reutilización por cuenta del autor. El archivo
`LICENSE.md` lo declara expresamente; el autor puede sustituirlo por la licencia
que decida conceder. Esto no cambia los derechos del dataset ni de dependencias.

El PDF y el proyecto Overleaf siguen siendo entregables académicos separados.
La aclaración de patches nnU-Net por fold figura en `docs/RESULTADOS.md`; aplíquela
al manuscrito antes de depositar una versión definitiva de la tesis.

La entrega local incluye una configuración de GitHub Actions. Sus pruebas CPU
se ejecutarán al publicarse; no se afirma que ya hayan corrido en GitHub.
