"""
Video Generator - Google Veo 3.1
สแกน Google Drive → extract frame → Image-to-Video ด้วย Veo 3.1 → Upload กลับ Drive

หมายเหตุ: Veo 3.1 รองรับสูงสุด 8 วินาที (ไม่มี 10 วิ)
"""

import os
import json
import time
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ==================== CONFIG ====================
SOURCE_FOLDER_ID   = os.environ.get("GDRIVE_SOURCE_FOLDER_ID")
OUTPUT_FOLDER_ID   = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
GDRIVE_CREDENTIALS = os.environ.get("GDRIVE_CREDENTIALS_JSON")

VEO_MODEL = "veo-3.1-generate-preview"

PROMPT = (
    "A beautiful young Thai woman, elegant and attractive, "
    "holding and showcasing the product with a bright smile, "
    "same product as original, same movements, "
    "clean bright background, natural studio lighting, "
    "vertical TikTok format, professional product showcase"
)

# ==================== LOGGING ====================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/run_{datetime.now().strftime('%Y%m%d')}.log"),
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

def list_videos_in_folder(service, folder_id):
    log.info("กำลังสแกน source folder...")
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'video/' and trashed=false",
        fields="files(id, name, size)"
    ).execute()
    files = results.get("files", [])
    log.info(f"พบวิดีโอ {len(files)} ไฟล์")
    return files

def download_video(service, file_id, file_name, tmp_dir):
    log.info(f"Downloading: {file_name}")
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(tmp_dir, file_name)
    with open(file_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    log.info(f"Downloaded: {file_name}")
    return file_path

def upload_video_to_drive(service, file_path, folder_id, file_name):
    log.info(f"Uploading to Drive: {file_name}")
    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
    file = service.files().create(
        body=file_metadata, media_body=media, fields="id, name", supportsAllDrives=True
    ).execute()
    log.info(f"Uploaded: {file['name']}")
    return file

# ==================== VIDEO PROCESSING ====================
def extract_frame(video_path, output_path, second=1):
    """Extract frame จากวิดีโอด้วย ffmpeg"""
    log.info(f"Extracting frame from {video_path}...")
    cmd = ["ffmpeg", "-i", video_path, "-ss", str(second), "-frames:v", "1", "-q:v", "2", output_path, "-y"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffmpeg error: {result.stderr}")
    log.info(f"Frame extracted: {output_path}")
    return output_path

# ==================== VEO 3.1 ====================
def generate_video_from_image(image_path, output_path):
    """Generate วิดีโอจาก image ด้วย Veo 3.1"""
    log.info(f"Generating video with Veo 3.1...")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # โหลด image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image = types.Image(image_bytes=image_bytes, mime_type="image/jpeg")

    # ส่ง request generate video
    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=PROMPT,
        image=image,
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",      # vertical สำหรับ TikTok/Reels
            duration_seconds="8",     # สูงสุด 8 วิ (Veo 3.1 limit)
            number_of_videos=1,
            resolution="720p",
        )
    )

    log.info(f"Operation started: {operation.name}")

    # Poll จนเสร็จ (อาจใช้เวลา 30 วิ - 6 นาที)
    while not operation.done:
        log.info("  Waiting for Veo 3.1... (polling every 15s)")
        time.sleep(15)
        operation = client.operations.get(operation)

    # ดึงผลลัพธ์
    if not operation.response or not operation.response.generated_videos:
        raise Exception("No video generated in response")

    video = operation.response.generated_videos[0]
    client.files.download(file=video.video)
    video.video.save(output_path)
    log.info(f"Video saved: {output_path}")
    return output_path

# ==================== MAIN ====================
def main():
    log.info("=" * 50)
    log.info(f"Veo 3.1 Video Generator เริ่มทำงาน: {datetime.now()}")
    log.info("=" * 50)

    service = get_drive_service()
    source_videos = list_videos_in_folder(service, SOURCE_FOLDER_ID)

    if not source_videos:
        log.warning("ไม่พบวิดีโอใน source folder")
        return

    success_count = 0
    fail_count = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for video in source_videos:
            video_name = video["name"]
            video_id   = video["id"]
            log.info(f"\n{'─'*40}")
            log.info(f"Processing: {video_name}")

            try:
                # 1. Download จาก Google Drive
                source_path = download_video(service, video_id, video_name, tmp_dir)

                # 2. Extract frame
                frame_path = os.path.join(tmp_dir, f"{Path(video_name).stem}_frame.jpg")
                extract_frame(source_path, frame_path, second=1)

                # 3. Generate วิดีโอด้วย Veo 3.1
                output_name = f"veo_{Path(video_name).stem}_{datetime.now().strftime('%Y%m%d')}.mp4"
                output_path = os.path.join(tmp_dir, output_name)
                generate_video_from_image(frame_path, output_path)

                # 4. Upload ขึ้น Google Drive
                upload_video_to_drive(service, output_path, OUTPUT_FOLDER_ID, output_name)

                success_count += 1
                log.info(f"สำเร็จ: {video_name} → {output_name}")

            except Exception as e:
                fail_count += 1
                log.error(f"Error: {video_name}: {str(e)}")
                continue

    log.info(f"\n{'='*50}")
    log.info(f"เสร็จสิ้น! สำเร็จ: {success_count} | ล้มเหลว: {fail_count}")
    log.info(f"{'='*50}")

if __name__ == "__main__":
    main()
