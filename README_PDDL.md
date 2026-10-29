# SAM-6D con los CAD y las fotos de PDDL

El punto de entrada es `pddl_pipeline.py`. La explicación y la preparación
común están en [SAM6D/pipeline](../../SAM6D/pipeline/README.md).
Los comandos siguientes parten de `/home/raca/projects`.

## Entorno CPU instalado

En este workspace se usa `.venvs/assembly-runtime`. Desde `/home/raca/projects`:

```bash
bash SAM6D/pipeline/run_runtime.sh sam doctor
bash SAM6D/pipeline/run_runtime.sh sam segment --frame control_board
bash SAM6D/pipeline/run_runtime.sh export
```

Se han ejecutado el renderizador y el ISM con los pesos oficiales SAM ViT-H y
DINOv2 ViT-L/14. Las plantillas de esta ejecución son de 256 px y 16 muestras
por píxel, conservando las 42 orientaciones. Los pesos están en una caché
temporal en RAM; consultar la guía común para las rutas y la exportación.

## Entorno original, como alternativa

Utilizar el `SAM-6D/environment.yaml` oficial para crear un entorno dedicado.
No utilizar el Python base 3.14 para el código original, que especifica Python
3.9 y PyTorch 2.0. Conservar NumPy 1.x para esa pila de dependencias antigua;
una instalación actual puede requerir fijar también OpenCV a una versión
compatible con NumPy 1.x. La creación de este entorno completo no se ha podido
validar aquí.

Desde el entorno SAM-6D, los descargadores oficiales son:

```bash
cd SAM6D/SAM-6D/SAM-6D/Instance_Segmentation_Model
python download_sam.py
python download_dinov2.py
cd ../Pose_Estimation_Model
python download_sam6d-pem.py
```

Para PEM se necesita además CUDA y compilar `model/pointnet2/setup.py` desde
su directorio. El modelo también utiliza el checkpoint MAE de su extractor,
que el código oficial descarga cuando se construye. `doctor` comprueba los
pesos principales y la extensión, pero no certifica la compatibilidad binaria
de todo el entorno. BlenderProc necesita Blender y las dependencias del script
en su propio intérprete, incluidas NumPy, OpenCV y trimesh.

## Comandos

```bash
python SAM6D/SAM-6D/pddl_pipeline.py doctor
python SAM6D/SAM-6D/pddl_pipeline.py render
python SAM6D/SAM-6D/pddl_pipeline.py segment --frame control_board
python SAM6D/SAM-6D/pddl_pipeline.py pose --frame control_board
```

`render` genera las 42 vistas oficiales de cada CAD. El nuevo renderizador
guarda XYZ métricos en milímetros mediante profundidad y calibración del
render, para que PEM reciba las mismas unidades que las mallas. Un marcador
con hash del CAD y unidades impide reutilizar una renderización interrumpida
o de otra malla. Esta fase ya se ha ejecutado en Blender CPU; `audit_templates.py` verifica sus XYZ métricos.

`segment` genera propuestas SAM una sola vez por imagen y las compara
conjuntamente con todos los CAD seleccionados, mediante los métodos originales
de ISM: semántica, apariencia, geometría y NMS por objeto. No clasifica usando
el nombre de la foto. Admite varias instancias y cero detecciones.

`pose` pasa las detecciones de cada clase al PEM original con su CAD y sus
plantillas. Guarda `T_camera_from_sam_mm` y `T_camera_from_local_mm` para
conservar el origen del CAD original. No genera una pose cuando no hay
detecciones por encima del umbral.

Opciones comunes: `--manifest`, `--frame` repetible, `--part` repetible,
`--output`, `--min-score` y `--dry-run`. Un ejemplo que procesa una sola captura
pero mantiene todas las clases candidatas:

```bash
python SAM6D/SAM-6D/pddl_pipeline.py segment --frame mounted_front_housing --min-score 0.3
```

Se han corregido también dos mutaciones de dimensiones que hacían fallar casos
con una sola detección, la propuesta SAM vacía y el tipo numérico del umbral
de PEM. Las regresiones de dimensiones se han ejecutado con PyTorch CPU;
el ISM se ha ejecutado en CPU. PEM sigue pendiente del acceso CUDA/GPU.
