import os
import zipfile


def extract_dataset(zip_filename, extract_dir):
    """
    แตกไฟล์ .zip ไปยังโฟลเดอร์เป้าหมาย
    """
    os.makedirs(extract_dir, exist_ok=True)
    print(f"กำลังแตกไฟล์ {zip_filename}...")

    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print("✅ แตกไฟล์เสร็จสมบูรณ์")


def find_dir(root, target_name):
    """
    ค้นหาโฟลเดอร์ที่ชื่อ (หรือมีคำว่า) target_name อยู่ภายใน root อัตโนมัติ
    """
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if target_name.lower() in d.lower():
                return os.path.join(dirpath, d)
    return None


def main():
    # 1. กำหนดชื่อไฟล์และโฟลเดอร์
    zip_filename = "data/raw/hyper-kvasir-segmented-images.zip"
    raw_dir = "data/raw_dataset"

    # 2. แตกไฟล์ชุดข้อมูล
    if os.path.exists(zip_filename):
        extract_dataset(zip_filename, raw_dir)
    else:
        print(f"❌ ไม่พบไฟล์ {zip_filename} กรุณาตรวจสอบพาร์ทให้ถูกต้อง")
        return

    # 3. ค้นหาโฟลเดอร์ images และ masks
    images_dir = find_dir(raw_dir, "image")
    masks_dir = find_dir(raw_dir, "mask")

    if images_dir and masks_dir:
        print(f"📂 พบโฟลเดอร์ images ที่: {images_dir}")
        print(f"📂 พบโฟลเดอร์ masks ที่: {masks_dir}")
        print(f"🖼️ จำนวนภาพ: {len(os.listdir(images_dir))}")
        print(f"🎭 จำนวน mask: {len(os.listdir(masks_dir))}")
    else:
        print("❌ ไม่พบโฟลเดอร์ images หรือ masks กรุณาระบุ Path เอง")

    # TODO: เพิ่มโค้ดส่วนการแบ่งข้อมูล (Train 70% / Val 20% / Test 10%)
    # TODO: เพิ่มโค้ดการแปลง Binary Mask เป็น YOLO Polygon Format


if __name__ == "__main__":
    main()
import os
import zipfile

def extract_dataset(zip_filename, extract_dir):
    """
    แตกไฟล์ .zip ไปยังโฟลเดอร์เป้าหมาย
    """
    os.makedirs(extract_dir, exist_ok=True)
    print(f"กำลังแตกไฟล์ {zip_filename}...")
    
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print("✅ แตกไฟล์เสร็จสมบูรณ์")

def find_dir(root, target_name):
    """
    ค้นหาโฟลเดอร์ที่ชื่อ (หรือมีคำว่า) target_name อยู่ภายใน root อัตโนมัติ
    """
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if target_name.lower() in d.lower():
                return os.path.join(dirpath, d)
    return None

def main():
    # 1. กำหนดชื่อไฟล์และโฟลเดอร์
    zip_filename = "data/raw/hyper-kvasir-segmented-images.zip"
    raw_dir = "data/raw_dataset"
    
    # 2. แตกไฟล์ชุดข้อมูล
    if os.path.exists(zip_filename):
        extract_dataset(zip_filename, raw_dir)
    else:
        print(f"❌ ไม่พบไฟล์ {zip_filename} กรุณาตรวจสอบพาร์ทให้ถูกต้อง")
        return

    # 3. ค้นหาโฟลเดอร์ images และ masks
    images_dir = find_dir(raw_dir, "image")
    masks_dir = find_dir(raw_dir, "mask")

    if images_dir and masks_dir:
        print(f"📂 พบโฟลเดอร์ images ที่: {images_dir}")
        print(f"📂 พบโฟลเดอร์ masks ที่: {masks_dir}")
        print(f"🖼️ จำนวนภาพ: {len(os.listdir(images_dir))}")
        print(f"🎭 จำนวน mask: {len(os.listdir(masks_dir))}")
    else:
        print("❌ ไม่พบโฟลเดอร์ images หรือ masks กรุณาระบุ Path เอง")
        
    # TODO: เพิ่มโค้ดส่วนการแบ่งข้อมูล (Train 70% / Val 20% / Test 10%) 
    # TODO: เพิ่มโค้ดการแปลง Binary Mask เป็น YOLO Polygon Format

if __name__ == "__main__":
    main()