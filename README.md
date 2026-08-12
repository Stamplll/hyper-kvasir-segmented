# Hyper-Kvasir Segmented — Polyp Detection (YOLO11-Seg)

โปรเจกต์เทรนโมเดล **YOLO11 Instance Segmentation** สำหรับตรวจจับและตีกรอบ (segment) ติ่งเนื้อ (polyp)
ในภาพส่องกล้องทางเดินอาหาร โดยใช้ชุดข้อมูล [Hyper-Kvasir Segmented](https://datasets.simula.no/hyper-kvasir/)

## โครงสร้างโปรเจกต์

```
hyper-kvasir-segmented/
│
├── data/
│   ├── raw/                # เก็บไฟล์ zip ต้นฉบับ (hyper-kvasir-segmented-images.zip)
│   └── processed/          # ข้อมูลที่แบ่ง train/val/test แล้ว พร้อม label (สร้างอัตโนมัติ)
│
├── notebooks/               # Jupyter Notebook สำหรับทดลองแนวทางก่อนแยกเป็นสคริปต์
│
├── src/
│   ├── data_prep.py         # แตกไฟล์ + แบ่งข้อมูล + แปลง mask เป็น YOLO polygon
│   ├── train.py             # เทรนโมเดล YOLO11n-seg
│   └── predict.py           # รัน inference บนภาพ/วิดีโอใหม่
│
├── weights/                 # โมเดลที่เทรนเสร็จแล้ว (.pt)
├── runs/                    # ผลลัพธ์การเทรน/predict (loss, confusion matrix, ภาพตัวอย่าง)
├── requirements.txt
└── README.md
```

## การติดตั้ง

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## วิธีใช้งาน

### 1. เตรียมข้อมูล

ดาวน์โหลด `hyper-kvasir-segmented-images.zip` มาวางไว้ที่ `data/raw/` แล้วรัน:

```bash
python src/data_prep.py
```

สคริปต์นี้จะ:
- แตกไฟล์ zip
- ค้นหาโฟลเดอร์ `images` / `masks` อัตโนมัติ
- แบ่งข้อมูลเป็น Train 70% / Val 20% / Test 10%
- แปลง binary mask เป็น YOLO polygon label (`.txt`)
- สร้าง `data/processed/dataset.yaml` สำหรับใช้เทรน

### 2. เทรนโมเดล

```bash
python src/train.py
```

ผลลัพธ์ (weights, กราฟ loss, confusion matrix) จะถูกบันทึกไว้ที่ `runs/segment/polyp_detection`
เมื่อพอใจกับผลแล้ว ให้คัดลอก weight ที่ดีที่สุด (`best.pt`) ไปไว้ที่โฟลเดอร์ `weights/`

### 3. รัน Inference

```bash
python src/predict.py --source path/to/image_or_video --weights weights/polyp_yolo11n_seg_best.pt --save
```

พารามิเตอร์หลัก:

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `--source` | (จำเป็นต้องระบุ) | ภาพ / วิดีโอ / โฟลเดอร์ / stream |
| `--weights` | `weights/polyp_yolo11n_seg_best.pt` | ไฟล์ weight ที่เทรนแล้ว |
| `--conf` | `0.79` | ค่า confidence threshold |
| `--imgsz` | `640` | ขนาดภาพที่ใช้ inference |
| `--save` | — | บันทึกภาพผลลัพธ์ลง `runs/predict` |

## หมายเหตุ

- `data/raw/`, `data/processed/`, `weights/`, `runs/` ถูก `.gitignore` ไว้ (เก็บเฉพาะ `.gitkeep`) เนื่องจากเป็นไฟล์ขนาดใหญ่ที่สร้างซ้ำได้
- โมเดลพื้นฐานที่ใช้คือ `yolo11n-seg.pt` (Nano) ปรับ `model = YOLO(...)` ใน `train.py` ได้หากต้องการโมเดลที่ใหญ่ขึ้น