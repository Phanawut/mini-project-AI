import ctypes
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = PROJECT_DIR / "video"
RUNS_DIR = PROJECT_DIR / "runs" / "detect"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.105
NMS_IOU_THRESHOLD = 0.7


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


def fit_frame_to_screen(frame, screen_width: int, screen_height: int):
    height, width = frame.shape[:2]
    scale = min(0.9 * screen_width / width, 0.8 * screen_height / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def main() -> None:
    video_paths = sorted(
        path
        for path in VIDEO_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ) if VIDEO_DIR.is_dir() else []
    if not video_paths:
        raise FileNotFoundError(f"ไม่พบไฟล์วิดีโอในโฟลเดอร์ {VIDEO_DIR}")

    model_path = find_latest_model()
    model = YOLO(str(model_path))
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    print(f"ใช้โมเดล: {model_path}")
    print(f"คลาสที่โมเดลรู้จัก: {model.names}")
    print(f"พบวิดีโอ {len(video_paths)} ไฟล์ กด 'q' เพื่อออก")

    try:
        for video_path in video_paths:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"เปิดวิดีโอไม่ได้: {video_path}")

            fps = capture.get(cv2.CAP_PROP_FPS)
            delay = max(1, round(1000 / fps)) if fps > 0 else 1
            window_name = f"Object Detection - {video_path.name}"
            print(f"กำลังตรวจจับ: {video_path.name}")

            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    result = model.predict(
                        source=frame,
                        imgsz=IMAGE_SIZE,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=NMS_IOU_THRESHOLD,
                        verbose=False,
                    )[0]
                    annotated_frame = fit_frame_to_screen(
                        result.plot(), screen_width, screen_height
                    )
                    cv2.imshow(window_name, annotated_frame)

                    if cv2.waitKey(delay) & 0xFF == ord("q"):
                        return
            finally:
                capture.release()
                cv2.destroyWindow(window_name)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
