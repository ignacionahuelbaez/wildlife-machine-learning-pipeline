# AGENTS.md: Guía para Agentes de IA

Este archivo contiene instrucciones directas y simplificadas para asistentes de IA (Gemini, Claude, OpenCode, Codex, etc.) que asistan en este repositorio.

---

## 🎯 Propósito del Repositorio
Mantener y entrenar un modelo de Computer Vision basado en **Ultralytics YOLO11s** para detectar fauna silvestre (inicialmente chanchos, con futuras ampliaciones a ciervos y más) desde imágenes aéreas de drones DJI, exportando el resultado a TFLite para Android.

---

## 🏗️ Organización y Ciclo de Vida de los Datos (Ubicación Automática)
La IA debe utilizar y ordenar los archivos estrictamente en las siguientes carpetas:
1. **`data/raw_videos/`**: Videos fuente originales (ej. `.mp4`). La IA debe buscar aquí cuando se soliciten extracciones de fotogramas.
2. **`data/unlabeled/`**: Fotogramas extraídos pendientes de etiquetado. Destino temporal antes de subir a CVAT.
3. **`data/dataset/`**: Dataset activo y validado (`images/train/`, `images/val/`, `labels/`, `data.yaml`).
4. **`models/trained/`**: Pesos entrenados activos (`best.pt`, `last.pt`).
5. **`models/exports/`**: Modelo exportado en `.tflite` para Android.

---

## 🗺️ Guía para Guiar al Usuario (CVAT y Expansión de Clases)
Cuando el usuario quiera agregar nuevos datos o especies (ej. ciervos):
1. **Extracción:** Indicar o realizar la extracción desde `data/raw_videos/` hacia `data/unlabeled/`.
2. **CVAT:** Guiar al usuario para subir los fotogramas a CVAT, anotar con las clases correctas (`0: chancho`, `1: ciervo`), exportar en formato YOLO y colocar el resultado en `data/dataset/`.
3. **Configuración (`data.yaml`):** Asegurar que `nc` y `names` en `data/dataset/data.yaml` coincidan con las clases.
4. **Entrenamiento:** Ejecutar `python3 scripts/validate_environment.py` y `python3 scripts/train_model.py`.

---

## 🛠️ Comandos Esenciales
- Validar entorno: `python3 scripts/validate_environment.py`
- Validar dataset: `python3 scripts/prepare_yolo_dataset.py`
- Entrenar: `python3 scripts/train_model.py`
