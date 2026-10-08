# Detector de Jabalíes y Fauna Silvestre con YOLO11s

Repositorio colaborativo de Machine Learning para el entrenamiento, mejora continua e integración móvil de modelos de detección de fauna (chanchos, ciervos, etc.) utilizando **Ultralytics YOLO11s**, diseñado para operación con drones DJI y despliegue en Android (**TensorLite/LiteRT**).

Este repositorio está diseñado para que tanto **usuarios humanos** como **agentes de IA** operen de forma fluida mediante una organización estricta del ciclo de vida de los datos.

---

## 🔄 Ciclo de Vida de los Datos y Organización Automática (Para IA y Humanos)

La estructura del repositorio está diseñada para que cualquier agente de IA sepa exactamente dónde ubicar, procesar y buscar cada archivo:

1. **`data/raw_videos/` (Videos Crudos):**
   - **Qué va aquí:** Videos originales capturados por el dron (ej. `chanchos100%.mp4`).
   - **Propósito:** Almacenar la fuente primaria antes de ser procesada. La IA sabe que debe buscar aquí cuando se le pida extraer nuevos fotogramas.
2. **`data/unlabeled/` (Fotogramas sin Anotar):**
   - **Qué va aquí:** Fotogramas extraídos de los videos crudos que aún no han pasado por CVAT.
   - **Propósito:** Área de tránsito previa al etiquetado manual o asistido.
3. **`data/dataset/` (Dataset Activo y Validado):**
   - **Qué va aquí:** Imágenes (`images/train/`, `images/val/`), etiquetas YOLO (`labels/`) y la configuración (`data.yaml`).
   - **Propósito:** Es el dataset final listo para alimentar los entrenamientos. Cuando se agregan nuevas especies (ej. ciervos), se actualiza aquí y se ajusta `nc` / `names` en `data.yaml`.
4. **`models/trained/` y `models/exports/`:**
   - **`models/trained/`:** Contiene los pesos activos (`best.pt`, `last.pt`) del modelo actual.
   - **`models/exports/`:** Contiene el modelo exportado a `.tflite` listo para Android.

---

## 📖 Guía de Uso Paso a Paso para el Usuario

### Paso 1: Ingesta de Nuevos Videos
Coloca tus nuevos videos aéreos del dron en:
```bash
data/raw_videos/
```

### Paso 2: Extracción y Preparación (Asistido por IA)
Pídele a tu agente de IA que extraiga los fotogramas del video en `data/raw_videos/` hacia `data/unlabeled/`.

### Paso 3: Anotación en CVAT
1. Sube los fotogramas de `data/unlabeled/` a CVAT.
2. Dibuja los *bounding boxes* respetando las clases en `data.yaml` (ej. `0: chancho`, `1: ciervo`).
3. Exporta el dataset desde CVAT en formato **YOLO 1.1 / YOLOv8**.
4. Coloca el resultado en `data/dataset/` (`images/` y `labels/`).

### Paso 4: Validar Entorno y Dataset
Ejecuta el script de validación para verificar que todo esté en orden:
```bash
python3 scripts/validate_environment.py
```

### Paso 5: Entrenar el Modelo
Ejecuta el script de entrenamiento para actualizar los pesos del modelo activo:
```bash
python3 scripts/train_model.py
```
Los nuevos pesos se guardarán en `models/trained/best.pt`.

### Paso 6: Exportación a Android
Exporta tu modelo entrenado a formato TFLite y colócalo en `models/exports/` para integrarlo en tu app de Android.
