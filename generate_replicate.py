"""
Video Generator - Replicate (Wan 2.1 Image-to-Video)
อ่าน prom.txt + รูปภาพจาก Google Drive → Generate video → Upload กลับ Google Drive

รูปแบบไฟล์ใน Google Drive folder:
    prom.txt   ← prompt แต่ละบรรทัด ขึ้นด้วยเลข
    1.jpg      ← รูปสินค้าสำหรับ prompt 1
    2.jpg      ← รูปสินค้าสำหรับ prompt 2
"""

import os
import io
import json
import time
import base64
import logging
import requests
import replicate
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ==================== CONFIG ====================
GDRIVE_PROMPT_FOLDER_ID = os.environ.get("GDRIVE_PROMPT_FOLDER_ID")
GDRIVE_OUTPUT_FOLDER_ID = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID")
REPLICATE_API_TOKEN     = os.environ.get("REPLICATE_API_TOKEN")
GDRIVE_CREDENTIALS      = os.environ.get("GDRIVE_CREDENTIALS_JSON")

PROMPT_FILENAME  = "prom.txt"
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]

# Wan 2.1 image-to-video model
REPLICATE_MODEL = "wavespeedai/wan-2.1-i2v-480p"

# ==================== LOGGING ====================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/replicate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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
    log.info(f"กำลังอ่าน '{PROMPT_FILENAME}' จาก Google Drive...")
    data, _ = download_file_bytes(service, folder_id, PROMPT_FILENAME)
    if data is None:
        raise FileNotFoundError(f"ไม่พบไฟล์ '{PROMPT_FILENAME}'")
    content = data.decode("utf-8")
    log.info(f"อ่านไฟล์สำเร็จ ({len(content)} ตัวอักษร)")
    return content

def download_image(service, folder_id, num):
    for ext in IMAGE_EXTENSIONS:
        filename = f"{num}.{ext}"
        data, mime = download_file_bytes(service, folder_id, filename)
        if data is not None:
            log.info(f"โหลดรูป: {filename}")
            return data, mime, ext
    log.warning(f"ไม่พบรูปภาพสำหรับ prompt [{num}]")
    return None, None, None

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

# ==================== REPLICATE ====================
def generate_video(prompt_text: str, image_bytes: bytes, mime_type: str, ext: str, output_path: str):
    """Generate วิดีโอด้วย Replicate Wan 2.1"""
    log.info("กำลัง Generate ด้วย Replicate Wan 2.1...")

    # แปลง image เป็น data URI
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_uri = f"data:{mime_type};base64,{b64}"

    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

    output = replicate.run(
        REPLICATE_MODEL,
        input={
            "image": image_uri,
            "prompt": prompt_text,
            "num_frames": 161,      # ~10 วิ ที่ 16fps
            "guidance_scale": 6,
            "num_inference_steps": 30,
        }
    )

    # output เป็น URL ของวิดีโอ
    video_url = str(output)
    log.info(f"Video URL: {video_url}")

    # Download วิดีโอ
    log.info("Downloading video...")
    resp = requests.get(video_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info(f"บันทึก: {output_path}")
    return output_path

# ==================== MAIN ====================
def main():
    log.info("=" * 55)
    log.info(f"🎬 Replicate Generator: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    service = get_drive_service()
    content = read_prompt_file(service, GDRIVE_PROMPT_FOLDER_ID)
    prompts = parse_prompts(content)

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

        output_name = f"video_r_{timestamp}_p{num}.mp4"
        output_path = f"/tmp/{output_name}"

        try:
            image_bytes, mime_type, ext = download_image(service, GDRIVE_PROMPT_FOLDER_ID, num)

            if image_bytes is None:
                log.error(f"❌ ไม่มีรูปสำหรับ prompt [{num}] — ข้ามไป")
                fail_count += 1
                continue

            generate_video(text, image_bytes, mime_type, ext, output_path)
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
