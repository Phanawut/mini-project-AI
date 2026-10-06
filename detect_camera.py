from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs" / "detect"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.105
NMS_IOU_THRESHOLD = 0.7


def find_latest_model() -> Path:
    checkpoints = [
        path
        for path in RUNS_DIR.glob("*/weights/best.pt")
        if path.is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"ไม่พบโมเดลที่เทรนแล้วใน {RUNS_DIR}\\<ชื่อรอบเทรน>\\weights\\best.pt "
            "กรุณารัน python train_model.py ให้เสร็จก่อน"
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    model_path = find_latest_model()
    model = YOLO(str(model_path))
    print(f"ใช้โมเดล: {model_path}")
    print(f"คลาสที่โมเดลรู้จัก: {model.names}")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("เปิดกล้องหมายเลข 0 ไม่ได้ กรุณาตรวจสอบการเชื่อมต่อกล้อง")

    print("เริ่มการทำงานของกล้อง... กดปุ่ม 'q' ที่หน้าต่างกล้องเพื่อออก")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("อ่านภาพจากกล้องไม่ได้")

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                verbose=False,
            )
            annotated_frame = results[0].plot()
            cv2.imshow("Custom Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
