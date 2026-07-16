from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
    APIRouter
)

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import base64
from PIL import Image, ImageEnhance, ImageFilter
import io
import json
import re
import requests
from typing import Optional

# =========================================================
# RTSP + WEBSOCKET
# =========================================================
import cv2
import threading
import asyncio
import time
import os
from urllib.parse import unquote

# =========================================================
# FFMPEG LOW LATENCY
# =========================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;500000"
)

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(title="Smart OCR API")

router = APIRouter()

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# CONFIG
# =========================================================
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "qwen2.5vl:latest"
)

TIMEOUT = int(
    os.getenv(
        "TIMEOUT",
        "180"
    )
)

session = requests.Session()

# =========================================================
# GLOBAL CAMERA STREAMS
# =========================================================
CAMERA_STREAMS = {}

# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def encode_image(image: Image.Image):

    buffer = io.BytesIO()

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = ImageEnhance.Contrast(image).enhance(1.8)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = ImageEnhance.Brightness(image).enhance(1.1)
    image = image.filter(ImageFilter.SHARPEN)

    image.thumbnail((600, 600), Image.Resampling.LANCZOS)

    image.save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True
    )

    return base64.b64encode(buffer.getvalue()).decode()

# =========================================================
# OCR EXTRACTION
# =========================================================
def extract_with_qwen(front_img: Image.Image):

    images = [encode_image(front_img)]

    prompt = """
You are a highly accurate document OCR expert.

Read the image carefully and extract information exactly as printed.

Return ONLY a valid JSON object with these exact keys:

{
  "user_name": "full name as printed or null",
  "document_type": "PAN Card / Aadhaar Card / Voter ID / Other or null",
  "document_number": "the main ID number or null",
  "date_of_birth": "DD/MM/YYYY or DD-MM-YYYY or null",
  "address": "full address if clearly visible or null",
  "mobile_number": "10-digit mobile number or null"
}

Strict Rules:
- Extract ONLY text you can clearly read
- Never guess
- If blurry use null
- Output must be pure JSON only
"""

    try:

        response = session.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "images": images,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 400,
                    "top_k": 20,
                    "top_p": 0.95,
                    "repeat_penalty": 1.1
                }
            },
            timeout=TIMEOUT
        )

        ollama_json = response.json()

        output = ollama_json.get("response", "").strip()

        if not output:
            return {"error": "Empty response"}, ""

        start = output.find("{")
        end = output.rfind("}") + 1

        if start != -1 and end > start:

            json_str = output[start:end]

            parsed = json.loads(json_str)

            return parsed, output

        return {"error": "No valid JSON"}, output

    except Exception as e:

        print(f"Ollama Error: {e}")

        return {"error": str(e)}, ""

# =========================================================
# REGEX FALLBACK
# =========================================================
def extract_with_regex(text: str):

    pan = re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text)

    aadhaar = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)

    epic = re.search(r'\b[A-Z]{3}\d{7}\b', text)

    dob = re.search(r'\b\d{2}[-/]\d{2}[-/]\d{4}\b', text)

    doc_num = (
        pan.group(0)
        if pan else (
            aadhaar.group(0)
            if aadhaar else (
                epic.group(0)
                if epic else None
            )
        )
    )

    return {
        "document_number": doc_num,
        "date_of_birth": dob.group(0) if dob else None
    }

# =========================================================
# OCR API
# =========================================================
@app.post("/upload")
async def upload_document(

    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)

):

    try:

        if file:

            contents = await file.read()

            image = Image.open(io.BytesIO(contents))

        elif image_base64:

            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]

            image_bytes = base64.b64decode(image_base64)

            image = Image.open(io.BytesIO(image_bytes))

        else:

            raise HTTPException(
                status_code=400,
                detail="No image provided"
            )

        ai_result, raw_output = extract_with_qwen(image)

        regex_result = extract_with_regex(raw_output)

        if ai_result.get("document_number") is None:
            ai_result["document_number"] = regex_result["document_number"]

        if ai_result.get("date_of_birth") is None:
            ai_result["date_of_birth"] = regex_result["date_of_birth"]

        return JSONResponse(content={
            "status": "success",
            "data": ai_result,
            "raw_output": raw_output[:1000]
        })

    except Exception as e:

        print(f"Upload Error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================================
# RTSP STREAMER
# =========================================================
class RTSPStreamer:

    def __init__(self, rtsp_url):

        self.rtsp_url = rtsp_url

        self.cap = None

        self.last_frame = None

        self.running = False

        self.lock = threading.Lock()

        self.thread = None

        self.clients = 0

        self.reconnect_delay = 2

    # =====================================================
    # CONNECT
    # =====================================================
    def connect(self):

        try:

            # =============================================
            # RELEASE OLD CONNECTION SAFELY
            # =============================================
            try:
                if self.cap:
                    self.cap.release()
            except:
                pass

            self.cap = None

            print(f"Connecting RTSP: {self.rtsp_url}")

            # =============================================
            # IMPORTANT:
            # DO NOT USE cv2.CAP_FFMPEG
            # =============================================
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # =============================================
            # LOW BUFFER
            # =============================================
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # =============================================
            # LOWER FPS FOR STABILITY
            # =============================================
            self.cap.set(cv2.CAP_PROP_FPS, 20)

            # =============================================
            # WIDTH
            # =============================================
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

            # =============================================
            # HEIGHT
            # =============================================
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # =============================================
            # CHECK CONNECTION
            # =============================================
            if not self.cap.isOpened():

                print(f"RTSP Failed: {self.rtsp_url}")

                return False

            print(f"RTSP Connected")

            return True

        except Exception as e:

            print(f"RTSP Connect Error: {e}")

            return False

    # =====================================================
    # START THREAD
    # =====================================================
    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )

        self.thread.start()

        print("Capture Thread Started")

    # =====================================================
    # CAPTURE LOOP
    # =====================================================
    def capture_loop(self):

        while self.running:

            try:

                # =========================================
                # CONNECT IF NEEDED
                # =========================================
                if self.cap is None or not self.cap.isOpened():

                    connected = self.connect()

                    if not connected:

                        time.sleep(self.reconnect_delay)

                        continue

                # =========================================
                # THREAD SAFE FRAME READ
                # =========================================
                with self.lock:

                    ret, frame = self.cap.read()

                # =========================================
                # FAILED FRAME
                # =========================================
                if not ret or frame is None:

                    print("Frame Read Failed")

                    try:
                        if self.cap:
                            self.cap.release()
                    except:
                        pass

                    self.cap = None

                    time.sleep(1)

                    continue

                # =========================================
                # SAVE FRAME
                # =========================================
                with self.lock:

                    self.last_frame = frame.copy()

            except Exception as e:

                print(f"Capture Error: {e}")

                try:
                    if self.cap:
                        self.cap.release()
                except:
                    pass

                self.cap = None

                time.sleep(1)

    # =====================================================
    # GET FRAME
    # =====================================================
    async def get_frame_base64(self):

        try:

            with self.lock:

                if self.last_frame is None:
                    return None

                frame = self.last_frame.copy()

            # =============================================
            # JPEG ENCODE
            # =============================================
            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 65]
            )

            if not success:
                return None

            return base64.b64encode(
                buffer
            ).decode()

        except Exception as e:

            print(f"Frame Encode Error: {e}")

            return None

    # =====================================================
    # STOP
    # =====================================================
    def stop(self):

        print("Stopping Stream...")

        self.running = False

        time.sleep(0.5)

        try:

            if self.cap:

                self.cap.release()

        except Exception as e:

            print(f"Release Error: {e}")

        self.cap = None

        self.last_frame = None

        print("Stream Stopped")

# =========================================================
# WEBSOCKET ENDPOINT
# =========================================================
@router.websocket("/ocr/ws")
async def ocr_websocket(
    websocket: WebSocket,
    rtsp_url: str = Query(...)
):

    await websocket.accept()

    rtsp_url = unquote(rtsp_url)

    print(f"Client Connected: {rtsp_url}")

    # =====================================================
    # SHARED STREAM
    # =====================================================
    if rtsp_url not in CAMERA_STREAMS:

        CAMERA_STREAMS[rtsp_url] = RTSPStreamer(rtsp_url)

        CAMERA_STREAMS[rtsp_url].start()

    streamer = CAMERA_STREAMS[rtsp_url]

    streamer.clients += 1

    print(f"Clients: {streamer.clients}")

    try:

        while True:

            # =============================================
            # CLIENT DISCONNECTED
            # =============================================
            if websocket.client_state.name != "CONNECTED":

                print("WebSocket Closed")

                break

            frame_b64 = await streamer.get_frame_base64()

            if frame_b64:

                await websocket.send_json({
                    "frame": frame_b64
                })

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:

        print("Client Disconnected")

    except Exception as e:

        print(f"WebSocket Error: {e}")

    finally:

        streamer.clients -= 1

        print(f"Remaining Clients: {streamer.clients}")

        # ================================================
        # AUTO CLEANUP
        # ================================================
        if streamer.clients <= 0:

            streamer.stop()

            del CAMERA_STREAMS[rtsp_url]

            print("Stream Removed")

# =========================================================
# INCLUDE ROUTER
# =========================================================
app.include_router(router)

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():

    return {
        "status": "running",
        "service": "OCR WebSocket RTSP API"
    }

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8713,
        reload=True,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )