# Clasificación de Sismicidad del volcán Galeras

Codigos para la clasificación de sismicidad (VT, LP, TRE, TOR) del volcán Galeras utilizando aprendizaje en contexto con TabPFN. Desarrollado como parte del proyecto de grado en la Universidad de Narino.

---

## 1. Instalación y Requisitos Previos

⚠️ **Importante - Version de Python:** Este proyecto fue desarrollado y probado exhaustivamente en **Python 3.10.9**. Se requiere utilizar la rama **3.10.x** para evitar problemas de compatibilidad y dependencias rotas con modelos avanzados como TabPFN.

Para ejecutar este proyecto de forma segura, utiliza un entorno virtual:

1. **Crear y activar el entorno virtual, usando la versión de python 3.10 para este proyecto:**
   ```
   python -3.10 -m venv nombre_del_entorno
   ```
   *Activa el entorno virtual para trabajar y correr los codigos.*
   
3. **Instalar las dependencias exactas:**
   Con el entorno activado, ejecuta el siguiente comando para instalar las versiones precisas requeridas:
   ```
   pip install -r requirements.txt
   ```
   *(Nota: El archivo instala automaticamente los paquetes criticos como `numpy`, `pandas`, `scipy`, `scikit-learn`, `obspy`, `librosa`, `tabpfn==6.4.1`, `matplotlib`, `pyarrow` y `fastparquet`).*

---

## 2. Codigos Principales y Pipeline del Proyecto

El repositorio esta disenado para ejecutarse de manera secuencial. A continuacion se describen los scripts principales que conforman el nucleo de la extracción y modelado.

*(Nota: De este bloque principal, el unico codigo que requiere configuracion manual de rutas de datos es `config_extract.py`).*

*   **`config_extract.py`**: Archivo maestro que centraliza los hiperparametros de procesamiento: frecuencia de muestreo (100Hz), parametros de los filtros (0.7Hz highpass), tamano de ventanas, umbrales de coda, fracciones de particion (80% contexto), y la lista de caracteristicas estadisticas y espectrales seleccionadas.
     * `INPUT_ROOT = # Dataset access path` (Ruta a la carpeta del dataset. Ejemplo: `r\"C:\Users\Nombre\OneDrive\Escritorio\Dataset"`).
  
*   **`extract_features.py`**: Script principal de procesamiento. Recorre el dataset leyendo los archivos `miniSEED`, limpia la señal, extrae las caracteristicas (temporales, frecuenciales, MFCC), divide el conjunto en particiones estratificadas y guarda los DataFrames resultantes en formato `.parquet`.
*   **`config_Transformer.py`**: Archivo de configuracion de hiperparametros para TabPFN. Define la semilla de reproducibilidad, tamano maximo de memoria, y el factor de peso de clases (`alpha`).   
  *(Nota: Modificar el valor de `alpha` a los valores 0.5 y 1, permite reproducir el experimento 4 de la investigación).*

*   **`Transformer.py`**: Codigo central de la investigacion. Emplea el aprendizaje en contexto de la arquitectura TabPFN, cargando el set de contexto a la memoria en tiempo de ejecucion para inferir sobre el conjunto de prueba.
   *(Nota: Si se requiere ejectura el Script de forma individual puede hacerlo de la forma `Transformer.py --partition k`, siendo k=1, 2, 3, 4. ).*

---

## 3. Codigos Secundarios y Configuracion de Rutas

Los codigos secundarios son para visualizacion, validacion y aumento de datos. Antes de ejecutar los scripts, es obligatorio configurar las rutas absolutas de tus carpetas locales en la cabecera de los siguientes scripts.

### A. Analisis y Visualizacion Exploratoria
*   **`dataset_viewer.py`**: Script para realizar el análisis exploratorio de datos (EDA)
    *   `SISMOS_DIR = # Dataset access path` (Ruta a la carpeta del dataset. Ejemplo: `r\"C:\Users\Nombre\OneDrive\Escritorio\Dataset"`).
    *   `OUTPUT_DIR = # Figure saving path` (Carpeta destino para las imagenes generadas).


### B. Validacion e Importancia
*   **`Random_forest.py`**: Entrena iterativamente un modelo basado en arboles para cada particion, estableciendo una metrica de linea base (baseline) y validando el aporte individual de cada caracteristica a la clasificacion. *El script es completamente dinamico y genera automaticamente sus rutas de salida basadas en el directorio de ejecucion actual, por lo que no necesita configurar rutas.*

### C. Aumento de Datos (Clase TOR)
*   **`data_augmentation.py`**: Script para generar datos aumentados del tipo de sismo TOR.
    *   `SISMOS_ROOT = # Dataset access path` (Ruta a la carpeta del dataset. Ejemplo: `r\"C:\Users\Nombre\OneDrive\Escritorio\Dataset"`).
    *   `OUTPUT_DIR = # Save path`
        > ⚠️ **Nota importante sobre el Output:** El guardado debe estar ubicado estrictamente en la misma ruta base de tu dataset original, dentro de la subcarpeta de la clase minoritaria a la que se le hizo el aumento. 
        > **Ejemplo correcto:** Si tu dataset esta en la carpeta `Dataset`, la salida debe ser `r\"C:\Users\Nombre\OneDrive\Escritorio\Dataset\TOR"`.

*   **`viewer_augmentation.py`**: Se utiliza para comparar la señal original del sismo TOR con su variante aumentada.
    *   `SISMOS_ROOT = # Dataset access path` (Ruta a la carpeta del dataset. Ejemplo: `r\"C:\Users\Nombre\OneDrive\Escritorio\Dataset"`).
    *   `OUTPUT_FIGS = # Figure saving path` (Directorio destino para las figuras comparativas).

---

## 4. Interfaz de usuario (Menú Principal)

Para facilitar la ejecución de todos los códigos mencionados anteriormente y evitar posibles problemas de rutas, se ha creado un script principal llamado **`main.py`** en la raíz del proyecto. Este archivo funciona como un lanzador interactivo.

Aunque puedes correr cada script de manera individual, **se recomienda usar el menú principal** para ejecutar cualquier código de las carpetas principales o secundarias de forma segura y centralizada.

**Para ejecutarlo:**
Simplemente abre una terminal en la raíz del proyecto y ejecuta:

```
python main.py
```

Esto abrirá un menú interactivo donde podrás seleccionar la carpeta y el script específico que deseas correr, sin necesidad de navegar manualmente entre los directorios en tu consola.
