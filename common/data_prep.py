"""
data_prep.py
เตรียมข้อมูลชุด Hyper-Kvasir Segmented สำหรับเทรนโมเดล YOLO11-Segmentation

ขั้นตอนการทำงาน:
1. แตกไฟล์ .zip ต้นฉบับ
2. ค้นหาโฟลเดอร์ images / masks อัตโนมัติ
3. แบ่งข้อมูลเป็น Train/Val/Test (70/20/10)
4. แปลง Binary Mask -> YOLO Polygon format (.txt)
5. สร้างไฟล์ dataset.yaml สำหรับใช้กับ ultralytics (train.py)
"""

import os
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml  

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
CLASS_NAMES = ["polyp"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def extract_dataset(zip_filename: str, extract_dir: str) -> None:
    """แตกไฟล์ .zip ไปยังโฟลเดอร์เป้าหมาย"""
    os.makedirs(extract_dir, exist_ok=True)
    print(f"กำลังแตกไฟล์ {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("✅ แตกไฟล์เสร็จสมบูรณ์")


def find_dir(root: str, target_name: str):
    """ค้นหาโฟลเดอร์ที่ชื่อ (หรือมีคำว่า) target_name อยู่ภายใน root อัตโนมัติ"""
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        for d in dirnames:
            if target_name.lower() in d.lower():
                return os.path.join(dirpath, d)
    return None


def mask_to_yolo_polygons(mask_path: str, img_w: int, img_h: int, class_id: int = 0):
    """
    แปลง binary mask (ขาว-ดำ) เป็น polygon รูปแบบ YOLO segmentation
    คืนค่า list ของบรรทัด "class_id x1 y1 x2 y2 ... xn yn" (พิกัด normalize 0-1)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 20:  # ข้าม noise เล็ก ๆ
            continue

        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        coords = []
        for point in approx:
            x, y = point[0]
            coords.append(x / img_w)
            coords.append(y / img_h)

        line = f"{class_id} " + " ".join(f"{c:.6f}" for c in coords)
        lines.append(line)

    return lines


def split_dataset(image_files: list, ratios: dict, seed: int = RANDOM_SEED):
    """สุ่มแบ่งรายชื่อไฟล์เป็น train/val/test ตามสัดส่วนที่กำหนด"""
    files = image_files.copy()
    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def find_matching_mask(masks_dir: str, stem: str):
    """หาไฟล์ mask ที่ชื่อ (stem) ตรงกับภาพ โดยไม่สนนามสกุล"""
    for ext in IMG_EXTS:
        candidate = os.path.join(masks_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_split(split_name: str, filenames: list, images_dir: str,
                   masks_dir: str, output_root: str):
    """คัดลอกภาพ + สร้างไฟล์ label .txt สำหรับ split หนึ่ง ๆ (train/val/test)"""
    img_out_dir = Path(output_root) / split_name / "images"
    lbl_out_dir = Path(output_root) / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for filename in filenames:
        stem = Path(filename).stem
        src_img_path = os.path.join(images_dir, filename)

        mask_path = find_matching_mask(masks_dir, stem)
        if mask_path is None:
            print(f"⚠️ ไม่พบ mask สำหรับ {filename} ข้ามไฟล์นี้")
            skipped += 1
            continue

        img = cv2.imread(src_img_path)
        if img is None:
            print(f"⚠️ อ่านภาพ {filename} ไม่ได้ ข้ามไฟล์นี้")
            skipped += 1
            continue
        h, w = img.shape[:2]

        shutil.copy2(src_img_path, img_out_dir / filename)

        polygon_lines = mask_to_yolo_polygons(mask_path, w, h)
        label_path = lbl_out_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(polygon_lines))

    done = len(filenames) - skipped
    print(f"✅ {split_name}: {done}/{len(filenames)} ภาพ -> {img_out_dir}")


def write_dataset_yaml(output_root: str):
    """สร้างไฟล์ dataset.yaml สำหรับให้ train.py เรียกใช้"""
    data = {
        "path": os.path.abspath(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = Path(output_root) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"✅ สร้างไฟล์ config: {yaml_path}")


def main():
    zip_filename = "data/raw/hyper-kvasir-segmented-images.zip"
    raw_dir = "data/raw_dataset"
    processed_dir = "data/processed"

    # 1. แตกไฟล์ชุดข้อมูล
    if os.path.exists(zip_filename):
        extract_dataset(zip_filename, raw_dir)
    else:
        print(f"❌ ไม่พบไฟล์ {zip_filename} กรุณาตรวจสอบพาธให้ถูกต้อง")
        return

    # 2. ค้นหาโฟลเดอร์ images และ masks
    images_dir = find_dir(raw_dir, "image")
    masks_dir = find_dir(raw_dir, "mask")

    if not (images_dir and masks_dir):
        print("❌ ไม่พบโฟลเดอร์ images หรือ masks กรุณาระบุ Path เอง")
        return

    print(f"📂 พบโฟลเดอร์ images ที่: {images_dir}")
    print(f"📂 พบโฟลเดอร์ masks ที่: {masks_dir}")

    image_files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)
    )
    print(f"🖼️ จำนวนภาพทั้งหมด: {len(image_files)}")

    if not image_files:
        print("❌ ไม่พบไฟล์ภาพในโฟลเดอร์ images")
        return

    # 3. แบ่งข้อมูล Train/Val/Test
    splits = split_dataset(image_files, SPLIT_RATIOS)
    print(
        f"📊 แบ่งข้อมูล -> train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )

    # 4. คัดลอกภาพ + แปลง mask เป็น YOLO polygon label
    for split_name, filenames in splits.items():
        process_split(split_name, filenames, images_dir, masks_dir, processed_dir)

    # 5. สร้างไฟล์ dataset.yaml
    write_dataset_yaml(processed_dir)

    print("\n🎉 เตรียมข้อมูลเสร็จสมบูรณ์! พร้อมสำหรับรัน src/train.py")


if __name__ == "__main__":
    main()