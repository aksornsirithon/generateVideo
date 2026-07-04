"""
Video Generator - Google Veo 3.1 (Image-to-Video)
อ่าน prom.txt + รูปภาพจาก Google Drive → Generate video → Upload กลับ Google Drive

รูปแบบไฟล์ใน Google Drive folder:
    prom.txt   ← prompt แต่ละบรรทัด ขึ้นด้วยเลข
    1.jpg      ← รูปสินค้าสำหรับ prompt 1
    2.jpg      ← รูปสินค้าสำหรับ prompt 2

รูปแบบ prom.txt:
    1 สร้างวิดีโอ...หนุ่มหล่อ...
    2 สร้างวิดีโอ...สาวสวย...

รันทุกวัน ตี 2 (เวลาไทย) ผ่าน GitHub Actions
"""

import os
import io
import json
import time
import logging
from datetime import datetime
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ==================== CONFIG ====================
GDRIVE_PROMPT_FOLDER_ID = os.environ.get("GDRIVE_PROMPT_FOLDER_ID")
GDRIVE_OUTPUT_FOLDER_ID = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID")
GEMINI_API_KEY          = os.environ.get("GEMINI_API_KEY")
GDRIVE_CREDENTIALS      = os.environ.get("GDRIVE_CREDENTIALS_JSON")

PROMPT_FILENAME  = "prom.txt"
VEO_MODEL        = "veo-3.1-generate-preview"
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]

# ==================== LOGGING ====================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==================== GOOGLE DRIVE ====================
def get_drive_service():
    creds_dict = json.loads(GDRIVE_CREDENTIALS)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def download_file_bytes(service, folder_id, filename):
    """ค้นหาและ download ไฟล์จาก Google Drive คืนค่าเป็น bytes"""
    results = service.files().list(
        q=f"'{folder_id}' in parents and name='{filename}' and trashed=false",
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get("files", [])
    if not files:
        return None, None
    file_info = files[0]
    request = service.files().get_media(fileId=file_info["id"])
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue(), file_info.get("mimeType", "image/jpeg")

def read_prompt_file(service, folder_id):
    """อ่าน prom.txt จาก Google Drive"""
    log.info(f"กำลังอ่าน '{PROMPT_FILENAME}' จาก Google Drive...")
    data, _ = download_file_bytes(service, folder_id, PROMPT_FILENAME)
    if data is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ '{PROMPT_FILENAME}' ใน folder")
    content = data.decode("utf-8")
    log.info(f"อ่านไฟล์สำเร็จ ({len(content)} ตัวอักษร)")
    return content

def download_image(service, folder_id, num):
    """
    ค้นหารูปภาพที่ชื่อตรงกับเลข prompt เช่น 1.jpg, 1.png, 1.jpeg
    คืนค่า (bytes, mime_type) หรือ (None, None) ถ้าไม่พบ
    """
    for ext in IMAGE_EXTENSIONS:
        filename = f"{num}.{ext}"
        data, mime = download_file_bytes(service, folder_id, filename)
        if data is not None:
            log.info(f"โหลดรูป: {filename}")
            return data, mime
    log.warning(f"ไม่พบรูปภาพสำหรับ prompt [{num}] (ค้นหา: {num}.jpg/.png/.jpeg)")
    return None, None

def upload_to_drive(service, file_path, folder_id, file_name):
    log.info(f"Uploading: {file_name}")
    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
    file = service.files().create(
        body=file_metadata, media_body=media, fields="id, name", supportsAllDrives=True
    ).execute()
    log.info(f"✅ Upload สำเร็จ: {file['name']}")
    return file

# ==================== PARSE PROMPTS ====================
def parse_prompts(content: str) -> list[dict]:
    """แยก prompts จาก prom.txt — รูปแบบ: '1 ข้อความ...' """
    prompts = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            prompts.append({"num": int(parts[0]), "text": parts[1].strip()})
        else:
            prompts.append({"num": len(prompts) + 1, "text": line})

    log.info(f"พบ {len(prompts)} prompt")
    for p in prompts:
        log.info(f"  [{p['num']}] {p['text'][:70]}...")
    return prompts

# ==================== VEO 3.1 ====================
def generate_video(prompt_text: str, image_bytes: bytes, mime_type: str, output_path: str):
    """Generate วิดีโอด้วย Veo 3.1 image-to-video"""
    log.info("กำลัง Generate ด้วย Veo 3.1 (image-to-video)...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    # สร้าง image object จาก bytes
    image = types.Image(image_bytes=image_bytes, mime_type=mime_type)

    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt_text,
        image=image,                         # ส่งรูปสินค้าไปด้วย
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",             # แนวตั้ง TikTok/Reels/Shorts
            duration_seconds="8",            # สูงสุด 8 วิ (Veo 3.1 limit)
            number_of_videos=1,
            resolution="720p",
        )
    )

    log.info(f"Operation: {operation.name} | รอผล...")
    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
        log.info("  ยังรอ Veo 3.1...")

    if not operation.response or not operation.response.generated_videos:
        raise Exception("Veo 3.1 ไม่มีวิดีโอในผลลัพธ์")

    video = operation.response.generated_videos[0]
    client.files.download(file=video.video)
    video.video.save(output_path)
    log.info(f"บันทึก: {output_path}")
    return output_path

def generate_video_text_only(prompt_text: str, output_path: str):
    """Generate วิดีโอด้วย Veo 3.1 text-only (กรณีไม่มีรูป)"""
    log.info("ไม่พบรูป — Generate แบบ text-to-video แทน...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt_text,
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",
            duration_seconds="8",
            number_of_videos=1,
            resolution="720p",
        )
    )

    log.info(f"Operation: {operation.name} | รอผล...")
    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)
        log.info("  ยังรอ Veo 3.1...")

    if not operation.response or not operation.response.generated_videos:
        raise Exception("Veo 3.1 ไม่มีวิดีโอในผลลัพธ์")

    video = operation.response.generated_videos[0]
    client.files.download(file=video.video)
    video.video.save(output_path)
    log.info(f"บันทึก: {output_path}")
    return output_path

# ==================== MAIN ====================
def main():
    log.info("=" * 55)
    log.info(f"🎬 Veo 3.1 Generator: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    service   = get_drive_service()
    content   = read_prompt_file(service, GDRIVE_PROMPT_FOLDER_ID)
    prompts   = parse_prompts(content)

    if not prompts:
        log.warning("ไม่พบ prompt ในไฟล์")
        return

    success_count = 0
    fail_count    = 0
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")

    for p in prompts:
        num  = p["num"]
        text = p["text"]
        log.info(f"\n{'─'*45}")
        log.info(f"📝 Prompt [{num}]: {text[:80]}...")

        output_name = f"video_{timestamp}_p{num}.mp4"
        output_path = f"/tmp/{output_name}"

        try:
            # โหลดรูปที่ชื่อตรงกับเลข prompt (เช่น 1.jpg)
            image_bytes, mime_type = download_image(service, GDRIVE_PROMPT_FOLDER_ID, num)

            if image_bytes:
                # image-to-video (มีรูปสินค้า)
                generate_video(text, image_bytes, mime_type, output_path)
            else:
                # text-to-video (ไม่มีรูป)
                generate_video_text_only(text, output_path)

            upload_to_drive(service, output_path, GDRIVE_OUTPUT_FOLDER_ID, output_name)
            success_count += 1
            log.info(f"✅ สำเร็จ: {output_name}")

        except Exception as e:
            fail_count += 1
            log.error(f"❌ Error prompt [{num}]: {e}")
            continue

    log.info(f"\n{'='*55}")
    log.info(f"เสร็จสิ้น! ✅ สำเร็จ: {success_count} | ❌ ล้มเหลว: {fail_count}")
    log.info(f"{'='*55}")

if __name__ == "__main__":
    main()
