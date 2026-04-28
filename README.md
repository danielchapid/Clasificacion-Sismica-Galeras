# Clasificación de Sismicidad del Volcán Galeras

Código para la clasificación de sismicidad (VT, LP, TRE, TOR) del Volcán Galeras utilizando aprendizaje en contexto con TabPFN. Desarrollado como parte del proyecto de grado en la Universidad de Nariño.

---

## 1. ⚙️ Instalación y Requisitos Previos

⚠️ **Importante - Versión de Python:** Este proyecto fue desarrollado y probado exhaustivamente en **Python 3.10.9**. Se requiere utilizar la rama **3.10.x** para evitar problemas de compatibilidad y dependencias rotas con modelos avanzados como TabPFN.

Para ejecutar este proyecto de forma segura, utiliza un entorno virtual:

1. **Crear y activar el entorno virtual (forzando el uso de Python 3.10 en Windows):**
   ```powershell
   py -3.10 -m venv venv_galeras
   .\venv_galeras\Scripts\activate
Instalar las dependencias exactas:
Con el entorno activado, ejecuta el siguiente comando para instalar las versiones precisas requeridas:

PowerShell
pip install -r requirements.txt
(Nota: El archivo instala automáticamente los paquetes críticos como numpy, pandas, scipy, scikit-learn, obspy, librosa, tabpfn==6.4.1, matplotlib, pyarrow y fastparquet).

2. 🚀 Códigos Principales y Pipeline del Proyecto
El repositorio está diseñado para ejecutarse de manera secuencial. A continuación se describen los 5 scripts principales que conforman el núcleo de la extracción y modelado.

(Nota: De este bloque principal, el único código que requiere configuración manual de rutas de datos es config_extract.py).

config_extract.py: Archivo maestro que centraliza los hiperparámetros de procesamiento: frecuencia de muestreo (100Hz), parámetros de los filtros (0.7Hz highpass), tamaño de ventanas, umbrales de coda, fracciones de partición (80% contexto), y la lista de características estadísticas y espectrales seleccionadas.

extract_features.py: Script principal de procesamiento. Recorre el dataset leyendo los archivos miniSEED, limpia la señal, extrae las características (temporales, frecuenciales, MFCC), divide el conjunto en particiones estratificadas y guarda los DataFrames resultantes en formato .parquet.

Random_forest.py: Entrena iterativamente un modelo basado en árboles para cada partición, estableciendo una métrica de línea base (baseline) y validando el aporte individual de cada característica a la clasificación. Este script es dinámico y genera automáticamente sus rutas de salida basadas en el directorio de ejecución actual.

config_Transformer.py: Archivo de configuración de hiperparámetros para TabPFN. Define la semilla de reproducibilidad, tamaño máximo de memoria, y el factor de peso de clases (alpha). Modificar el valor de alpha permite reproducir el Experimento 4 de la investigación.

Transformer.py: Código central de la investigación. Emplea el aprendizaje en contexto de la arquitectura TabPFN, cargando el set de entrenamiento a la memoria en tiempo de ejecución para inferir sobre el conjunto de prueba.

Ejecución: Se debe evaluar partición por partición usando el argumento --partition.

python Transformer.py --partition 1

3. 📂 Configuración de Rutas Locales (Paths)
Antes de ejecutar la extracción y los códigos auxiliares de visualización/aumento, es obligatorio configurar las rutas absolutas de tus carpetas locales en la cabecera de los siguientes scripts.

Ejemplo general de ruta en Windows: r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"

A. Ruta del Pipeline Principal
config_extract.py

INPUT_ROOT = # Input dataset directory path (Ruta a la carpeta principal de tu dataset).

B. Rutas de Análisis y Visualización Exploratoria
dataset_viewer.py

SISMOS_DIR = # Dataset access path (Ruta original de los datos).

OUTPUT_DIR = # Figure saving path (Carpeta destino para las imágenes generadas).

C. Rutas de Aumento de Datos (Clase TOR)
data_augmentation.py

SISMOS_ROOT = # Dataset access path (Ruta original de los datos).

OUTPUT_DIR = # Save path

⚠️ Nota importante sobre el Output: El guardado debe estar ubicado estrictamente en la misma ruta base de tu dataset original, dentro de la subcarpeta de la clase minoritaria a la que se le hizo el aumento. Ejemplo correcto: Si tu dataset está en Sismos, la salida debe ser r"C:\Users\Daniel\OneDrive\Escritorio\Sismos\TOR".

viewer_augmentation.py

SISMOS_ROOT = # Dataset access path (Ruta original de los datos).

AUGMENTED_DIR = # Path to augmented data (Ruta específica a la subcarpeta de tornillos generados).

OUTPUT_FIGS = # Figure saving path (Directorio destino para las figuras comparativas).
