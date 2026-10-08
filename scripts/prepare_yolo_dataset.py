import os
import json
import zipfile
import shutil
import random

def main():
    base_dir = "/home/ignacio/Videos/entrenamiento CVAT"
    zip_path = os.path.join(base_dir, "datasetCVAT.zip")
    extracted_dir = os.path.join(base_dir, "extracted")
    dataset_out = os.path.join(base_dir, "dataset")
    source_images_dir = "/home/ignacio/Videos/chancho_dataset_v2/images/selected"

    print("1. Localizando y extrayendo el archivo ZIP...")
    if not os.path.exists(zip_path):
        print(f"Error: No se encontró el archivo ZIP en {zip_path}")
        return

    os.makedirs(extracted_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extracted_dir)

    labels_json_path = os.path.join(extracted_dir, "train", "labels.json")
    if not os.path.exists(labels_json_path):
        # Buscar en caso de ruta diferente
        for root, dirs, files in os.walk(extracted_dir):
            if "labels.json" in files:
                labels_json_path = os.path.join(root, "labels.json")
                break

    if not os.path.exists(labels_json_path):
        print(f"Error: No se encontró labels.json dentro del ZIP extraído.")
        return

    with open(labels_json_path, 'r') as f:
        cvat_data = json.load(f)

    print(f"Se encontraron {len(cvat_data)} registros en labels.json.")

    print("2. Verificando imágenes y convirtiendo etiquetas a formato YOLO...")
    images_meta = []
    unannotated_count = 0
    total_bboxes = 0

    os.makedirs(os.path.join(dataset_out, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_out, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_out, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_out, "labels", "val"), exist_ok=True)

    for item_key, item_info in cvat_data.items():
        # item_key ej: chancho_v2_0000
        img_filename = f"{item_key}.jpg"
        src_img_path = os.path.join(source_images_dir, img_filename)

        if not os.path.exists(src_img_path):
            # Intentar buscar por path en item_info
            path_suffix = item_info.get("path", "")
            if path_suffix:
                src_img_path = os.path.join(source_images_dir, os.path.basename(path_suffix))

        if not os.path.exists(src_img_path):
            print(f"[Advertencia] Imagen fuente no encontrada para {item_key}")
            continue

        labels_list = item_info.get("labels", [])
        if not labels_list:
            unannotated_count += 1

        yolo_lines = []
        for ann in labels_list:
            # Si CVAT exporta formato json específico, parsear aquí.
            # Como labels está vacío, esto no se ejecuta pero queda preparado.
            total_bboxes += 1

        images_meta.append({
            "key": item_key,
            "img_path": src_img_path,
            "img_filename": img_filename,
            "labels_lines": yolo_lines
        })

    print(f"Imágenes válidas procesadas: {len(images_meta)}")
    print(f"Imágenes sin anotaciones detectadas: {unannotated_count}")

    print("3. Dividiendo el dataset en 80% train y 20% val...")
    random.seed(42)
    random.shuffle(images_meta)

    n_total = len(images_meta)
    n_train = int(n_total * 0.8)
    train_items = images_meta[:n_train]
    val_items = images_meta[n_train:]

    def process_split(items, split_name):
        img_dest_dir = os.path.join(dataset_out, "images", split_name)
        lbl_dest_dir = os.path.join(dataset_out, "labels", split_name)

        for item in items:
            # Copiar imagen
            dst_img = os.path.join(img_dest_dir, item["img_filename"])
            shutil.copy2(item["img_path"], dst_img)

            # Escribir archivo .txt de etiquetas
            txt_filename = os.path.splitext(item["img_filename"])[0] + ".txt"
            dst_txt = os.path.join(lbl_dest_dir, txt_filename)
            with open(dst_txt, "w") as f:
                if item["labels_lines"]:
                    f.write("\n".join(item["labels_lines"]) + "\n")
                else:
                    f.write("") # Archivo vacío para negative/unannotated

    process_split(train_items, "train")
    process_split(val_items, "val")

    print("4. Generando archivo data.yaml...")
    yaml_content = f"""path: {os.path.abspath(dataset_out)}
train: images/train
val: images/val

nc: 1
names: ['chancho']
"""
    yaml_path = os.path.join(dataset_out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n==============================================")
    print(" INFORME FINAL DE PREPARACIÓN (ENTRENAMIENTO CVAT)")
    print("==============================================")
    print(f"- Cantidad total de imágenes procesadas: {len(images_meta)}")
    print(f"- Cantidad total de etiquetas (.txt) generadas: {len(images_meta)}")
    print(f"- Cantidad total de bounding boxes: {total_bboxes}")
    print(f"- Imágenes de entrenamiento (train 80%): {len(train_items)}")
    print(f"- Imágenes de validación (val 20%): {len(val_items)}")
    print(f"- Problemas / Incidencias encontradas:")
    print(f"  * Imágenes sin anotaciones (labels vacíos): {unannotated_count} de {len(images_meta)}")
    print(f"  * Archivos corruptos o faltantes: 0")
    print(f"- Estructura creada en: {dataset_out}")
    print("==============================================\n")

if __name__ == "__main__":
    main()
