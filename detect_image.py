import ctypes
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "image"
RUNS_DIR = PROJECT_DIR / "runs" / "detect"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.105
NMS_IOU_THRESHOLD = 0.7


def fit_image_to_screen(image, screen_width: int, screen_height: int):
    height, width = image.shape[:2]
    scale = min(0.9 * screen_width / width, 0.8 * screen_height / height, 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def find_latest_model() -> Path:
    checkpoints = [
        path for path in RUNS_DIR.glob("*/weights/best.pt") if path.is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"ไม่พบโมเดลที่เทรนแล้วใน {RUNS_DIR}\\<ชื่อรอบเทรน>\\weights\\best.pt "
            "กรุณารัน python train_model.py ให้เสร็จก่อน"
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if IMAGE_DIR.is_dir() else []
    if not image_paths:
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพในโฟลเดอร์ {IMAGE_DIR}")

    model_path = find_latest_model()
    model = YOLO(str(model_path))
    print(f"ใช้โมเดล: {model_path}")
    print(f"คลาสที่โมเดลรู้จัก: {model.names}")
    print(f"พบภาพ {len(image_paths)} ไฟล์ กดปุ่มใดก็ได้เพื่อดูภาพถัดไป หรือกด 'q' เพื่อออก")

    try:
        for image_path in image_paths:
            results = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                verbose=False,
            )
            result = results[0]
            print(f"{image_path.name}: ตรวจพบ {len(result.boxes)} วัตถุ")
            screen_width = ctypes.windll.user32.GetSystemMetrics(0)
            screen_height = ctypes.windll.user32.GetSystemMetrics(1)
            display_image = fit_image_to_screen(
                result.plot(), screen_width, screen_height
            )
            window_name = f"Object Detection - {image_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
