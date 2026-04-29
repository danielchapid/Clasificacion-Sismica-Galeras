"""
Menú Principal — Tesis Galeras
-------------------------------
Lanzador interactivo para todos los scripts del proyecto.
Ejecutar desde la raíz del proyecto.
"""
import subprocess
import sys
import os


# ==============================================================================
# RUTAS
# ==============================================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDERS = {
    "1": {
        "name": "Códigos principales",
        "path": os.path.join(ROOT_DIR, "Codigos principales"),
        "scripts": [
            ("config_extract.py",     "Configuración de extracción de características"),
            ("config_Transformer.py",  "Configuración del Transformer TabPFN"),
            ("extract_features.py",    "Pipeline de extracción de características"),
            ("Transformer.py",         "Clasificación y evaluación TabPFN"),
        ],
    },
    "2": {
        "name": "Códigos secundarios",
        "path": os.path.join(ROOT_DIR, "Codigo secundarios"),
        "scripts": [
            ("dataset_viewer.py",        "Análisis exploratorio de datos (EDA)"),
            ("Random_forest.py",         "Ranking de importancia de características (Random Forest)"),
            ("data_augmentation.py",     "Aumentación de datos de Tornillo"),
            ("viewer_augmentation.py",   "Visualización de señales aumentadas"),
        ],
    },
}


# ==============================================================================
# AYUDAS DE VISUALIZACIÓN
# ==============================================================================

W = 58

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def header(title: str):
    print(f"\n  {'='*W}")
    print(f"  {title}")
    print(f"  {'='*W}\n")


def pause():
    print()
    input("  Presione Enter para volver al menú...")


# ==============================================================================
# MENÚ
# ==============================================================================

def show_main_menu():
    clear_screen()
    header("MENÚ PRINCIPAL")
    print("  Seleccione la carpeta que desea abrir:\n")
    for key, folder in FOLDERS.items():
        print(f"    {key}. {folder['name']}")
    print(f"\n    0. Salir")
    print(f"\n  {'─'*W}")
    return input("\n  Opción: ").strip()


def show_scripts_menu(folder_key: str):
    folder = FOLDERS[folder_key]
    clear_screen()
    header(f"{folder['name'].upper()}")
    print("  Seleccione el script que desea ejecutar:\n")
    for i, (filename, description) in enumerate(folder["scripts"], start=1):
        print(f"    {i}. {filename:<28s} — {description}")
    print(f"\n    0. Volver")
    print(f"\n  {'─'*W}")
    return input("\n  Opción: ").strip()


def run_script(folder_key: str, script_index: int):
    """Lanza el script seleccionado como subproceso."""
    folder  = FOLDERS[folder_key]
    script  = folder["scripts"][script_index][0]
    cwd     = folder["path"]
    filepath = os.path.join(cwd, script)

    if not os.path.isfile(filepath):
        print(f"\n  [ERROR] Archivo no encontrado: {filepath}")
        pause()
        return

    # Para Transformer.py, preguntar qué partición evaluar
    extra_args = []
    if script == "Transformer.py":
        print(f"\n  Transformer.py requiere un número de partición.")
        part = input("  Ingrese la partición (1-4): ").strip()
        if part not in ("1", "2", "3", "4"):
            print("  [ERROR] Número de partición inválido.")
            pause()
            return
        extra_args = ["--partition", part]

    print(f"\n  {'─'*W}")
    print(f"  Ejecutando: {script}")
    print(f"  Directorio: {cwd}")
    print(f"  {'─'*W}\n")

    try:
        subprocess.run(
            [sys.executable, script] + extra_args,
            cwd=cwd,
        )
    except KeyboardInterrupt:
        print("\n\n  [INFO] Ejecución interrumpida por el usuario.")
    except Exception as e:
        print(f"\n  [ERROR] {e}")

    pause()


# ==============================================================================
# BUCLE PRINCIPAL
# ==============================================================================

def main():
    while True:
        choice = show_main_menu()

        if choice == "0":
            clear_screen()
            break

        if choice not in FOLDERS:
            continue

        while True:
            script_choice = show_scripts_menu(choice)

            if script_choice == "0":
                break

            n_scripts = len(FOLDERS[choice]["scripts"])
            if script_choice.isdigit() and 1 <= int(script_choice) <= n_scripts:
                run_script(choice, int(script_choice) - 1)
            # entrada inválida → volver a mostrar el menú de scripts


if __name__ == "__main__":
    main()
