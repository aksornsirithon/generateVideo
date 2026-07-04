"""
Video Generator - Google Veo 3.1
อ่าน prom.txt จาก Google Drive → Generate video ต่อ prompt → Upload กลับ Google Drive

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
GDRIVE_PROMPT_FOLDER_ID = os.environ.get("GDRIVE_PROMPT_FOLDER_ID")   # folder ที่มี prom.txt
GDRIVE_OUTPUT_FOLDER_ID = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID")   # folder เก็บวิดีโอผลลัพธ์
GEMINI_API_KEY          = os.environ.get("GEMINI_API_KEY")
GDRIVE_CREDENTIALS      = os.environ.get("GDRIVE_CREDENTIALS_JSON")

PROMPT_FILENAME = "prom.txt"
VEO_MODEL       = "veo-3.1-generate-preview"

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

def read_prompt_file_from_drive(service, folder_id, filename):
    """ค้นหาและอ่านไฟล์ prompt จาก Google Drive"""
    log.info(f"กำลังค้นหาไฟล์ '{filename}' ใน Google Drive...")
    results = service.files().list(
        q=f"'{folder_id}' in parents and name='{filename}' and trashed=false",
        fields="files(id, name)"
    ).execute()
    files = results.get("files", [])
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ '{filename}' ใน folder ที่กำหนด")

    file_id = files[0]["id"]
    log.info(f"พบไฟล์: {files[0]['name']} (ID: {file_id})")

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    content = buffer.getvalue().decode("utf-8")
    log.info(f"อ่านไฟล์สำเร็จ ({len(content)} ตัวอักษร)")
    return content

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
    """
    แยก prompt จากเนื้อหาไฟล์
    รองรับรูปแบบ:
        1 ข้อความ prompt แรก...
        2 ข้อความ prompt สอง...
    """
    prompts = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # บรรทัดที่ขึ้นด้วยตัวเลขตามด้วยช่องว่าง เช่น "1 ..." หรือ "2 ..."
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            num   = int(parts[0])
            text  = parts[1].strip()
            prompts.append({"num": num, "text": text})
        else:
            # ถ้าไม่มีเลขนำหน้า ให้ถือว่าเป็น prompt เดียว
            prompts.append({"num": len(prompts) + 1, "text": line})

    log.info(f"พบ {len(prompts)} prompt")
    for p in prompts:
        log.info(f"  [{p['num']}] {p['text'][:60]}...")
    return prompts

# ==================== VEO 3.1 ====================
def generate_video(prompt_text: str, output_path: str):
    """Generate วิดีโอด้วย Veo 3.1"""
    log.info("กำลัง Generate ด้วย Veo 3.1...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=prompt_text,
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",   # แนวตั้ง TikTok/Reels/Shorts
            duration_seconds="8",  # Veo 3.1 รองรับสูงสุด 8 วิ
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

    # 1. เชื่อมต่อ Google Drive
    service = get_drive_service()

    # 2. อ่าน prom.txt จาก Drive
    content = read_prompt_file_from_drive(service, GDRIVE_PROMPT_FOLDER_ID, PROMPT_FILENAME)

    # 3. แยก prompts
    prompts = parse_prompts(content)
    if not prompts:
        log.warning("ไม่พบ prompt ในไฟล์")
        return

    # 4. Generate วิดีโอแต่ละ prompt
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
            generate_video(text, output_path)
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
