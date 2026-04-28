# Clasificación de Sismicidad del Volcán Galeras

Código para la clasificación de sismicidad (VT, LP, TRE, TOR) del Volcán Galeras utilizando aprendizaje en contexto con TabPFN. Desarrollado como parte del proyecto de grado en la Universidad de Nariño.

---

## ⚙️ Instalación y Requisitos Previos

⚠️ **Importante - Versión de Python:** Este proyecto fue desarrollado y probado exhaustivamente en **Python 3.10**. Se recomienda estrictamente utilizar esta versión (o específicamente 3.10.9) para evitar problemas de compatibilidad y dependencias rotas con modelos avanzados como TabPFN.

Para ejecutar este proyecto de forma segura, utiliza un entorno virtual:

1. **Crear y activar el entorno virtual (usando Python 3.10):**
   ```powershell
   python -m venv venv_galeras
   venv_galeras\Scripts\activate
Instalar las dependencias exactas:
Con el entorno activado, ejecuta el siguiente comando para instalar las versiones precisas requeridas:

PowerShell
pip install -r requirements.txt
(Nota: El archivo instala automáticamente los paquetes críticos como numpy, pandas, scipy, scikit-learn, obspy, librosa, tabpfn==6.4.1, matplotlib, pyarrow y fastparquet).

📂 Configuración de Rutas Locales (Paths)
Antes de ejecutar los códigos, es obligatorio configurar las rutas absolutas donde están almacenados tus archivos .mseed locales y dónde deseas guardar los resultados. Debes abrir los siguientes scripts y modificar las variables en la cabecera.

Ejemplo de formato de ruta en Windows: r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"

Para Extracción y Visualización:
config_extract.py:

INPUT_ROOT = # Ruta a la carpeta principal de tu dataset

dataset_viewer.py (Análisis Exploratorio EDA):

SISMOS_DIR = # Ruta original de los datos

OUTPUT_DIR = # Carpeta destino para las imágenes generadas

Para Aumento de Datos (Clase TOR):
data_augmentation.py:

SISMOS_ROOT = # Ruta original de los datos

OUTPUT_DIR = # Ruta destino del aumento
(⚠️ Atención: La salida debe estar dentro de la misma ruta base original, en la carpeta específica de la clase minoritaria. Ej: r"...\Sismos\TOR").

viewer_augmentation.py:

SISMOS_ROOT = # Ruta original de los datos

AUGMENTED_DIR = # Ruta a la subcarpeta de tornillos sintéticos

OUTPUT_FIGS = # Directorio destino para las figuras comparativas

🚀 Pipeline del Proyecto y Códigos Principales
El proyecto está diseñado para ejecutarse de manera secuencial. A continuación se describe la función de los scripts nucleares de procesamiento y modelado:

1. Extracción de Características
config_extract.py: Archivo maestro que centraliza los hiperparámetros de procesamiento: frecuencia de muestreo (100Hz), parámetros de los filtros (0.7Hz highpass), tamaño de ventanas, umbrales de coda, fracciones de partición (80% contexto), y la lista de características estadísticas y espectrales seleccionadas.

extract_features.py: Script principal de procesamiento. Recorre el dataset leyendo los archivos miniSEED, limpia la señal, extrae las características (temporales, frecuenciales, MFCC), divide el conjunto en particiones estratificadas y guarda los DataFrames resultantes en formato .parquet.

2. Importancia de Características (Baseline)
Random_forest.py: Entrena iterativamente un modelo basado en árboles para cada partición, estableciendo una métrica de línea base (baseline) y validando el aporte individual de cada característica a la clasificación. Este script es dinámico y genera automáticamente sus rutas de salida basadas en el directorio de ejecución actual.

3. Modelado y Clasificación Avanzada
config_Transformer.py: Archivo de configuración de hiperparámetros para TabPFN. Define la semilla de reproducibilidad, tamaño máximo de memoria, y el factor de peso de clases (alpha). Nota: Modificar el valor de alpha permite reproducir el Experimento 4 de la investigación.

Transformer.py: Código central de la investigación. Emplea el aprendizaje en contexto de la arquitectura TabPFN, cargando el set de entrenamiento a la memoria en tiempo de ejecución para inferir sobre el conjunto de prueba.

Ejecución del Modelo:
El clasificador debe evaluarse partición por partición utilizando el argumento --partition k (donde k va de 1 a 4). Ejemplo en terminal:

PowerShell
python Transformer.py --partition 1
