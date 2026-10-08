import os
import time
from ultralytics import YOLO

def main():
    base_dir = "/home/ignacio/Videos/entrenamiento CVAT"
    yaml_path = os.path.join(base_dir, "dataset", "data.yaml")

    print("=== INICIANDO ENTRENAMIENTO YOLO11s (CPU) ===")
    print(f"Dataset YAML: {yaml_path}")
    print("Parámetros: model=yolo11s.pt, epochs=50, imgsz=640, batch=4, workers=0, device=cpu")

    start_time = time.time()

    model = YOLO("yolo11s.pt")

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=4,
        workers=0,
        device="cpu",
        project=os.path.join(base_dir, "runs"),
        name="chancho_yolo11s",
        exist_ok=True
    )

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{hours}h {minutes}m {seconds}s"

    best_model_path = os.path.join(base_dir, "runs", "chancho_yolo11s", "weights", "best.pt")
    last_model_path = os.path.join(base_dir, "runs", "chancho_yolo11s", "weights", "last.pt")

    print("\n==============================================")
    print(" INFORME FINAL DE ENTRENAMIENTO YOLO11s")
    print("==============================================")
    print(f"- Ruta de best.pt: {best_model_path}")
    print(f"- Ruta de last.pt: {last_model_path}")
    print(f"- mAP50: {results.box.map50:.4f}")
    print(f"- mAP50-95: {results.box.map:.4f}")
    print(f"- Precision: {results.box.p.mean():.4f}")
    print(f"- Recall: {results.box.r.mean():.4f}")
    print(f"- Tiempo total de entrenamiento: {time_str}")
    print("==============================================\n")

if __name__ == "__main__":
    main()
