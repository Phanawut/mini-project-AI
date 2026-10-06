import random
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_DIR / "train" / "images"
LABELS_DIR = PROJECT_DIR / "train" / "labels"
CONFIG_PATH = PROJECT_DIR / "data.yaml"
MODEL_PATH = PROJECT_DIR / "yolov8n.pt"
VALIDATION_RATIO = 0.2
SPLIT_SEED = 42
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def prepare_dataset(config_dir: Path) -> Path:
    images = sorted(
        (
            path
            for path in IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )
    if len(images) < 2:
        raise ValueError(f"ต้องมีภาพอย่างน้อย 2 ภาพใน {IMAGES_DIR}")

    missing_labels = [
        image.name for image in images if not (LABELS_DIR / f"{image.stem}.txt").is_file()
    ]
    if missing_labels:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ label สำหรับภาพ: {', '.join(missing_labels[:5])}"
        )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError(f"ไฟล์ {CONFIG_PATH} ต้องมีรายการคลาสใน 'names'")

    shuffled_images = images.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled_images)
    validation_count = max(
        1, min(len(images) - 1, round(len(images) * VALIDATION_RATIO))
    )
    validation_images = set(shuffled_images[:validation_count])

    for image in images:
        split = "valid" if image in validation_images else "train"
        image_dir = config_dir / split / "images"
        label_dir = config_dir / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(LABELS_DIR / f"{image.stem}.txt", label_dir / f"{image.stem}.txt")

    config["path"] = config_dir.as_posix()
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config.pop("test", None)
    dataset_config = config_dir / "data.yaml"
    dataset_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return dataset_config


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="yolo_dataset_") as temp_dir:
        dataset_config = prepare_dataset(Path(temp_dir))
        model = YOLO(str(MODEL_PATH))
        model.train(
            data=str(dataset_config),
            epochs=50,
            imgsz=640,
            seed=SPLIT_SEED,
            project=str(PROJECT_DIR / "runs" / "detect"),
        )
        print(f"การเทรนเสร็จสิ้น โมเดลที่ดีที่สุดอยู่ที่: {model.trainer.best}")


if __name__ == "__main__":
    main()
