import os
from ultralytics import YOLO


def train_model():
    """
    สคริปต์สำหรับรันเทรนโมเดล YOLO11 Segmentation สำหรับตรวจจับ Polyp
    """
    print("🚀 เริ่มต้นการเตรียมเทรนโมเดล YOLO11n-seg...")

    # โหลดโมเดล YOLO11 สำหรับงาน Instance Segmentation (Nano model)
    model = YOLO('yolo11n-seg.pt')

    # ระบุพาร์ทไปยังไฟล์ตั้งค่า Dataset (จะถูกสร้างในขั้นตอน data_prep)
    dataset_yaml_path = os.path.abspath('data/processed/dataset.yaml')

    if not os.path.exists(dataset_yaml_path):
        print(f"❌ ไม่พบไฟล์ {dataset_yaml_path} กรุณารัน data_prep.py ก่อนครับ")
        return

    # เริ่มกระบวนการเทรน
    print("⏳ กำลังเทรนโมเดล โปรดรอสักครู่...")
    results = model.train(
        data=dataset_yaml_path,
        epochs=100,               # จำนวนรอบการเทรน
        imgsz=640,                # ขนาดรูปภาพที่ใช้เทรน (640x640 เป็นมาตรฐาน)
        batch=16,                 # ขนาด Batch size (ปรับแต่งได้ตาม RAM ของ GPU)
        # ไม่ระบุ device -> ultralytics เลือกอัตโนมัติเอง (GPU ถ้ามี, ไม่มีก็ใช้ CPU)
        # หมายเหตุ: device='auto' ใช้ไม่ได้ ultralytics ไม่รู้จักค่านี้ ทำให้ error ทุกเครื่อง
        project='runs/segment',   # โฟลเดอร์หลักสำหรับเก็บผลลัพธ์
        name='polyp_detection',   # ชื่อโฟลเดอร์ย่อยของการรันครั้งนี้
        exist_ok=True             # อนุญาตให้บันทึกทับโฟลเดอร์เดิมถ้ามีอยู่แล้ว
    )

    print("✅ การเทรนเสร็จสมบูรณ์!")
    print("ผลลัพธ์และกราฟต่างๆ (เช่น Confusion Matrix, Loss curves) ถูกบันทึกไว้ที่: runs/segment/polyp_detection")


if __name__ == '__main__':
    train_model()