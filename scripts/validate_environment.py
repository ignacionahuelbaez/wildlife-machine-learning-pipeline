import os
import sys

def validate_env():
    print("=== VALIDACIÓN DE ENTORNO Y REPRODUCIBILIDAD ===")
    
    # 1. Versión de Python
    py_version = sys.version_info
    print(f"[*] Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("[!] Advertencia: Se recomienda Python 3.8 o superior.")
    else:
        print("[OK] Versión de Python compatible.")

    # 2. Dependencias instaladas
    required_pkgs = ["ultralytics", "torch", "cv2", "yaml", "PIL", "imagehash"]
    missing = []
    for pkg in required_pkgs:
        try:
            if pkg == "cv2":
                import cv2
            elif pkg == "yaml":
                import yaml
            elif pkg == "PIL":
                import PIL
            elif pkg == "imagehash":
                import imagehash
            elif pkg == "torch":
                import torch
            elif pkg == "ultralytics":
                import ultralytics
            print(f"  - {pkg}: INSTALADO")
        except ImportError:
            print(f"  - {pkg}: [FALTANTE]")
            missing.append(pkg)

    if missing:
        print(f"\n[!] Faltan dependencias: {missing}. Ejecute: pip install -r requirements.txt")
    else:
        print("[OK] Todas las dependencias principales están instaladas.")

    # 3. Existencia del dataset activo
    dataset_yaml = "data/dataset/data.yaml"
    if os.path.exists(dataset_yaml):
        print(f"[OK] Dataset activo encontrado en: {dataset_yaml}")
    else:
        print(f"[!] Dataset activo NO encontrado en: {dataset_yaml}")

    # 4. Existencia del modelo base
    pretrained_model = "models/pretrained/yolo11s.pt"
    if os.path.exists(pretrained_model):
        print(f"[OK] Modelo base encontrado en: {pretrained_model}")
    else:
        print(f"[!] Modelo base NO encontrado en: {pretrained_model}")

    # 5. Estructura correcta del repositorio
    expected_dirs = ["data/dataset", "data/raw_videos", "data/unlabeled", "scripts", "configs", "models/trained", "models/exports"]
    missing_dirs = [d for d in expected_dirs if not os.path.exists(d)]
    if missing_dirs:
        print(f"[!] Faltan directorios estructurales: {missing_dirs}")
    else:
        print("[OK] Estructura de directorios del repositorio correcta.")

    print("================================================")

if __name__ == "__main__":
    validate_env()
