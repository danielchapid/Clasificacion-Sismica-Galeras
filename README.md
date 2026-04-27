# Clasificacion-Sismica-Galeras

Código para la clasificación de sismicidad (VT, LP, TRE, TOR) del Volcán Galeras utilizando aprendizaje en contexto con TabPFN. Desarrollado para trabajo de grado en la Universidad de Nariño.

## Instalación y Requerimientos

Para ejecutar este proyecto, se recomienda encarecidamente utilizar un entorno virtual (venv) para evitar conflictos de versiones y librerías innecesarias.

1. **Crear y activar el entorno virtual:**
2. **Instalar las dependencias exactas:**
   Con el archivo `requirements.txt`, simplemente corre:
   ```powershell
   pip install -r requirements.txt
   ```
   *(Si lo haces manualmente, asegúrate de instalar las librerías principales: `numpy`, `pandas`, `scipy`, `scikit-learn`, `obspy`, `librosa`, `tabpfn`, `matplotlib`, `seaborn`, `tqdm`, `pyarrow`, `fastparquet`).* 

---

## Configuración de Rutas (Paths)

Varios scripts requieren que configures manualmente las rutas (paths) absolutas para leer los archivos sísmicos locales (`miniSEED`) y guardar los resultados. A continuación se detalla qué debes modificar en la cabecera de cada script antes de ejecutarlo.

**Ejemplo general de ruta en Windows:** `r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"`

### 1. `config_extract.py`
Define la carpeta donde se encuentra el conjunto de datos.
*   `INPUT_ROOT = # Input dataset directory path` (Ejemplo: `r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"`)

### 2. `data_augmentation.py`
Se encarga de aumentar los datos de la clase minoritaria (TOR).
*   `SISMOS_ROOT = # Dataset access path` (Ejemplo: `r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"`)
*   `OUTPUT_DIR = # Save path`
    > **Nota importante sobre el Output:** La ruta de guardado debe estar ubicada en la misma ruta base de tu dataset original, dentro de la clase TOR, la cual fue a la que se le aumento los datos. 
    > **Ejemplo correcto:** Si tu dataset está en `Sismos`, la salida debe ser `r"C:\Users\Daniel\OneDrive\Escritorio\Sismos\TOR"`.

### 3. `dataset_viewer.py`
Genera visualizaciones exploratorias (EDA) de los eventos sísmicos originales.
*   `SISMOS_DIR = # Input dataset directory` (Ruta original de los datos). ejemplo: r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"
*   `OUTPUT_DIR = # Output directory for generated images` (Carpeta donde se guardarán las figuras resultantes).

### 4. `viewer_augmentation.py`
Visualiza comparativamente las señales originales vs las señales sintéticas aumentadas.
*   `SISMOS_ROOT = # Dataset access path` (Ruta al dataset original). ejemplo: r"C:\Users\Daniel\OneDrive\Escritorio\Sismos"
*   `AUGMENTED_DIR = # Path to augmented data` (Ruta a la subcarpeta de tornillos generados, ej: `r"C:\Users\Daniel\OneDrive\Escritorio\Sismos\TOR"`).
*   `OUTPUT_FIGS = # Output directory for generated figures` (Directorio de guardado de la comparación).

---

## Descripción de los Códigos Principales

A continuación se describe la utilidad y el modo de ejecución de los códigos base del pipeline:

### Extracción de Características
*   **`config_extract.py`**: Archivo de configuración global que define frecuencias de muestreo (100Hz), parámetros de los filtros (0.7Hz highpass), variables de ventanas, umbrales de coda, fracciones para el dataset (80% contexto), y la lista de características seleccionadas a calcular.

*   **`extract_features.py`**: Recorre recursivamente las carpetas del dataset para leer archivos `miniSEED`. Limpia la señal y extrae características temporales, frecuenciales y MFCC. Finalmente, divide todo el conjunto de datos en particiones (contexto y prueba) y guarda los resultados en archivos `.parquet`.

### Modelado y Evaluación (Transformer)
*   **`config_Transformer.py`**: Archivo de configuración específico para TabPFN. Define la cantidad de estimadores, semilla de reproducibilidad, tamaño máximo de filas en memoria, y factor de peso de clases (`alpha`), al modificar este valor de alpha, se le dará reproducibilidad al Experimento 4 de la investigación.

*   **`Transformer.py`**: Es el código central que ejecuta la clasificación. Emplea el aprendizaje en contexto de TabPFN, pasándole el set de entrenamiento (80%) a la memoria en tiempo de ejecución e infiriendo sobre el conjunto de prueba.
    *   **Cómo ejecutarlo:** Se debe evaluar **partición por partición** usando el argumento `--partition k`, donde `k` es un valor del 1 al 4.
    *   **Ejemplo en terminal:**
        ```powershell
        python Transformer.py --partition 1
        ```

### Importancia de Características
*   **`Random_forest.py`** (Originalmente `feature_importance.py`): Entrena iterativamente un modelo basado en árboles en cada partición para validar cuáles características son las que más aportan a las decisiones de clasificación. *Este código es completamente dinámico y no requiere configurar rutas de entrada ni de salida*; crea la ruta basándose en el directorio actual.
