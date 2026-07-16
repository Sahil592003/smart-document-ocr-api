# 📄 Smart Document OCR API

A production-ready AI-powered Document OCR API built using **FastAPI**, **Qwen2.5-VL**, **Ollama**, **OpenCV**, and **WebSocket** technology. The system extracts structured information from identity documents while also supporting real-time RTSP video streaming.

---

## 📌 Overview

This project provides an enterprise-ready OCR solution capable of extracting structured information from identity documents using Vision Language Models (VLMs). It combines AI-based OCR with regex fallback validation and includes a built-in RTSP streaming service for real-time camera integration.

Supported document types include:

- Aadhaar Card
- PAN Card
- Voter ID
- Other Government Identity Documents

---

## 🚀 Features

- AI-Powered OCR using Qwen2.5-VL
- Intelligent Document Information Extraction
- Automatic JSON Output
- Regex Fallback Validation
- Image Enhancement & Preprocessing
- Base64 Image Support
- File Upload Support
- FastAPI REST API
- RTSP Camera Streaming
- WebSocket Live Streaming
- Low-Latency Video Processing
- Production-Ready Architecture

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Backend | FastAPI |
| Vision Language Model | Qwen2.5-VL |
| LLM Runtime | Ollama |
| Computer Vision | OpenCV |
| Image Processing | Pillow |
| Communication | WebSocket |
| Streaming | RTSP |
| HTTP Client | Requests |

---

## 📂 Project Structure

```text
smart-document-ocr-api/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
└── LICENSE
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Sahil592003/smart-document-ocr-api.git

cd smart-document-ocr-api
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Rename

```
.env.example
```

to

```
.env
```

Update

```
OLLAMA_BASE_URL
```

with your Ollama server URL.

---

## Run

```bash
python main.py
```

Server starts at

```
http://localhost:8713
```

Swagger Documentation

```
http://localhost:8713/docs
```

---

# 📡 API Endpoints

## OCR API

```
POST /upload
```

Supports:

- Image Upload
- Base64 Image Upload

Returns

```json
{
  "status":"success",
  "data":{
      "user_name":"John Doe",
      "document_type":"PAN Card",
      "document_number":"ABCDE1234F",
      "date_of_birth":"01/01/1995",
      "address":"Sample Address",
      "mobile_number":"9876543210"
  }
}
```

---

## Live RTSP Streaming

```
WebSocket

ws://localhost:8713/ocr/ws?rtsp_url=<RTSP_URL>
```

Returns

- Live Camera Frames
- Low Latency Streaming

---

# 🔄 OCR Workflow

```
Document Image
       │
       ▼
Image Preprocessing
       │
       ▼
Qwen2.5-VL OCR
       │
       ▼
JSON Extraction
       │
       ▼
Regex Validation
       │
       ▼
Structured Response
```

---

# 🚀 Performance Optimizations

- Vision Language Model OCR
- Automatic Image Enhancement
- JSON Validation
- Regex Fallback Extraction
- Multi-threaded RTSP Streaming
- WebSocket Communication
- Low-Latency Processing
- Automatic Camera Reconnection

---

# 🔒 Privacy

This repository does **not** include:

- Personal Identity Documents
- Customer Information
- Production Images
- Internal Server URLs
- API Credentials

Users can test the API using their own sample identity documents.

---

# 📌 Future Improvements

- Multi-language OCR
- Passport OCR
- Driving License OCR
- Face Verification
- QR Code Detection
- Document Classification
- Docker Support
- JWT Authentication
- Database Integration

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Sahil Ghadge**

AI/ML Engineer | Computer Vision Engineer | Generative AI Engineer

### Skills

- OCR
- Computer Vision
- Vision Language Models
- FastAPI
- Python
- OpenCV
- Qwen2.5-VL
- Ollama
- WebSocket
- RTSP
- Image Processing
- Generative AI

---

⭐ If you found this project useful, consider giving it a Star on GitHub.
