# Clasificacion de Sismicidad del Volcan Galeras

Codigos para la clasificacion de sismicidad (VT, LP, TRE, TOR) del Volcan Galeras utilizando aprendizaje en contexto con TabPFN. Desarrollado como parte del proyecto de grado en la Universidad de Narino.

---

## 1. ♩️ Instalacion y Requisitos Previos

⚠️ **Importante - Version de Python:** Este proyecto fue desarrollado y probado exhaustivamente en **Python 3.10.9**. Se requiere utilizar la rama **3.10.x** para evitar problemas de compatibilidad y dependencias rotas con modelos avanzados como TabPFN.

Para ejecutar este proyecto de forma segura, utiliza un entorno virtual:

1. **Crear y activar el entorno virtual (forzando el uso de Python 3.10 en Windows):**
   ```powershell
   python -3.10 -m venv venv_galeras
   .\venv_galeras\Scripts\activate
   ```

2. **Instalar las dependencias exactas:**
   Con el entorno activado, ejecuta el siguiente comando para instalar las versiones precisas requeridas:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Nota: El archivo instala automaticamente los paquetes criticos como `numpy`, `pandas`, `scipy`, `scikit-learn`, `obspy`, `librosa`, `tabpfn==6.4.1`, `matplotlib`, `pyarrow` y `fastparquet`).*

---

## 2. 🚀 Codigos Principales y Pipeline del Proyecto

El repositorio esta disenado para ejecutarse de manera secuencial. A continuacion se describen los scripts principales que conforman el nucleo de la extraccion y modelado.

*(Nota: De este bloque principal, el unico codigo que requiere configuracion manual de rutas de datos es `config_extract.py`).*

:   **`config_extract.py`**: Archivo maestro que centraliza los hiperparametros de procesamiento: frecuencia de muestreo (100Hz), parametros de los filtros (0.7Hz highpass), tamano de ventanas, umbrales de coda, fracciones de particion (80% contexto), y la lista de caracteristicas estadisticas y espectrales seleccionadas.
*   **`extract_features.py`**: Script principal de procesamiento. Recorre el dataset leyendo los archivos `miniSEED`, limpia la senal, extrae las caracteristicas (temporales, frecuenciales, MFCC), divide el conjunto en particiones estratificadas y guarda los DataFrames resultantes en formato `.parquet`.
*   **`config_Transformer.py`**: Archivo de configuracion de hiperparametros para TabPFN. Define la semilla de reproducibilidad, tamano maximo de memoria, y el factor de peso de clases (`alpha`). Modificar el valor de `alpha` permite reproducir el Experimento 4 de la investigacion.
*   **iTransformer.py`**: Codigo central de la investigacion. Emplea el aprendizaje en contexto de la arquitectura TabPFN, cargando el set de entrenamiento a la memoria en tiempo de ejecucion para inferir sobre el conjunto de prueba.
    *   **Ejecucion:** Se debe evaluar particion por particion usando el argumento `--partition`.
        ```powershell
        python Transformer.py --partition 1
        ```

---

## 3. 🤃 Codigos Secundarios y Configuracion de Rutas (Paths)

Los codigos secunfarios son para visualizacion, validacion y aumento de datos. Antes de ejecutar la extraccion (`config_extract.py`) y estos codigos, es obligatorio configurar las rutas absolutas de tus carpetas locales en la cabecera de los siguientes scripts.

**Ejemplo general de ruta en Windows:** `r\"C:\Users\Daniel\OneDrive\Escritorio\Sismos\"`

### A. Extraccion Principal
*   **`config_extract.py`**
    *   `INPUT_ROOT = # Input dataset directory path` (Ruta a la carpeta principal de tu dataset. Ejemplo: `r\"C:\Users\Daniel\OneDrive\Escritorio\Sismos\"`).

### B. Validacion e Importancia
*   **`Random_forest.py`** (Originalmente `feature_importance.py`): Entrena iterativamente un modelo basado en arboles para cada particion, estableciendo una metrica de linea base (baseline) y validando el aporte individual de cada caracteristica a la clasificacion. *El script es completamente dinamico y genera automaticamente sus rutas de salida basadas en el directorio de ejecucion actual, por lo que no necesita configurar rutas.*


### C. Analisis y Visualizacion Exploratoria
*   **`dataset_viewer.py`**
    *   `SISMOS_DIR = # Dataset access path` (Ruta original de los datos. Ejemplo: `r\"C:\Users\Daniel\OneDrive\Escritorio\Sismos\"`).
    *   `OUTPUT_DIR = # Figure saving path` (Carpeta destino para las imagenes generadas).

### D. Aumento de Datos (Clase TOR)
*   **`data_augmentation.py`**
    *   `SISMOS_ROOT = # Dataset access path` (Ruta original de los datos. Ejemplo: `r\"C:\Users\Daniel\OneDrive\Escritorio\Sismos\"`).
    *   `OUTPUT_DIR = # Save path`
        > ⚠️ **Nota importante sobre el Output:** El guardado debe estar ubicado estrictamente en la misma ruta base de tu dataset original, dentro de la subcarpeta de la clase minoritaria a la que se le hizo el aumento. 
        > **Ejemplo correcto:** Si tu dataset esta en `Sismos`, la salida debe ser `r\"C:\Users\Daniel\OneDrive\Escritorio\Sismos\TOR\"`.

*   **`viewer_augmentation.py`**
    *   `SISMOS_ROOT = # Dataset access path` (Ruta original de los datos. Ejemplo: `r\"C:\Users\Daniel\OneDrive\Escritorio\Sismos\"`).
    *   `AUGMENTED_DIR = # Path to augmented data` (Ruta especifica a la subcarpeta de tornillos generados. Ejemplo: `r\"C:\Users\Daniel\OneDrive\Escritorio\Sismos\TOR\"`.
    *   `OUTPUT_FIGS = # Figure saving path` (Directorio destino para las figuras comparativas).
