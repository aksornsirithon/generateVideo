"""
Video Generator - Kling AI API
สแกน Google Drive → extract frame → Image-to-Video ด้วย Kling AI → Upload กลับ Drive
"""

import os
import json
import time
import logging
import subprocess
import base64
import requests
import tempfile
import jwt  # PyJWT
from datetime import datetime
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ==================== CONFIG ====================
SOURCE_FOLDER_ID   = os.environ.get("GDRIVE_SOURCE_FOLDER_ID")
OUTPUT_FOLDER_ID   = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID")
KLING_ACCESS_KEY   = os.environ.get("KLING_ACCESS_KEY")
KLING_SECRET_KEY   = os.environ.get("KLING_SECRET_KEY")
GDRIVE_CREDENTIALS = os.environ.get("GDRIVE_CREDENTIALS_JSON")

KLING_API_BASE = "https://api.klingai.com"

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

# ==================== JWT AUTH ====================
def get_kling_token():
    """สร้าง JWT token สำหรับ Kling AI API"""
    now = int(time.time())
    payload = {
        "iss": KLING_ACCESS_KEY,
        "exp": now + 1800,  # หมดอายุใน 30 นาที
        "nbf": now - 5
    }
    token = jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256")
    return token

def get_kling_headers():
    return {
        "Authorization": f"Bearer {get_kling_token()}",
        "Content-Type": "application/json"
    }

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

def image_to_base64(image_path):
    """แปลงรูปเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ==================== KLING AI API ====================
def create_image_to_video_task(image_path):
    """สร้าง image-to-video task ใน Kling AI"""
    log.info("Creating Kling AI image-to-video task...")

    # แปลงรูปเป็น base64
    image_b64 = image_to_base64(image_path)

    payload = {
        "model_name": "kling-v1-6",
        "image": image_b64,
        "prompt": PROMPT,
        "negative_prompt": "blurry, low quality, distorted, ugly",
        "mode": "std",        # std = standard (ฟรี), pro = professional (เสีย credit มากกว่า)
        "duration": "10",     # 10 วินาที
        "aspect_ratio": "9:16"
    }

    resp = requests.post(
        f"{KLING_API_BASE}/v1/videos/image2video",
        headers=get_kling_headers(),
        json=payload
    )

    if resp.status_code != 200:
        raise Exception(f"Kling API error {resp.status_code}: {resp.text}")

    data = resp.json()
    if data.get("code") != 0:
        raise Exception(f"Kling API error: {data.get('message')}")

    task_id = data["data"]["task_id"]
    log.info(f"Task created: {task_id}")
    return task_id

def wait_for_task(task_id, timeout=600):
    """รอจนกว่า task จะเสร็จ"""
    log.info(f"Waiting for task {task_id}...")
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.get(
            f"{KLING_API_BASE}/v1/videos/image2video/{task_id}",
            headers=get_kling_headers()
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise Exception(f"API error: {data.get('message')}")

        task_data = data["data"]
        status = task_data.get("task_status")
        log.info(f"  Status: {status}")

        if status == "succeed":
            videos = task_data.get("task_result", {}).get("videos", [])
            if not videos:
                raise Exception("No video in result")
            return videos[0]["url"]

        elif status == "failed":
            msg = task_data.get("task_status_msg", "unknown error")
            raise Exception(f"Task failed: {msg}")

        time.sleep(15)

    raise Exception(f"Timeout after {timeout}s waiting for task {task_id}")

def download_video_from_url(url, output_path):
    """Download วิดีโอจาก URL"""
    log.info(f"Downloading generated video...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info(f"Saved: {output_path}")

# ==================== MAIN ====================
def main():
    log.info("=" * 50)
    log.info(f"Kling Video Generator เริ่มทำงาน: {datetime.now()}")
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

                # 2. Extract frame แรก
                frame_path = os.path.join(tmp_dir, f"{Path(video_name).stem}_frame.jpg")
                extract_frame(source_path, frame_path, second=1)

                # 3. สร้าง Kling task
                task_id = create_image_to_video_task(frame_path)

                # 4. รอผลลัพธ์
                video_url = wait_for_task(task_id)

                # 5. Download วิดีโอที่ generate แล้ว
                output_name = f"kling_{Path(video_name).stem}_{datetime.now().strftime('%Y%m%d')}.mp4"
                output_path = os.path.join(tmp_dir, output_name)
                download_video_from_url(video_url, output_path)

                # 6. Upload ขึ้น Google Drive
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
