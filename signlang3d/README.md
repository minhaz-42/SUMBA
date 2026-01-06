# 🤟 SUMBA — 3D Sign Language Translation Platform

<p align="center">
  <img src="docs/logo.png" alt="SUMBA Logo" width="180">
</p>

<p align="center">
  <strong>Capture • Train • Translate — Real-Time Sign Language to Text</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#supported-languages">Languages</a> •
  <a href="#api">API</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Django-4.2-green?logo=django" alt="Django">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Overview

SUMBA is a research-grade platform for **3D sign language translation**. Unlike traditional 2D video-based approaches, SUMBA captures **21 hand joint positions in 3D space** (X, Y, Z coordinates) using MediaPipe, enabling more accurate gesture recognition and translation.

### Why 3D Beats 2D

| 2D Video Limitations | 3D Skeletal Advantages |
|---------------------|------------------------|
| ❌ Depth ambiguity | ✅ Full spatial information (X, Y, Z) |
| ❌ Occlusion issues | ✅ View invariance |
| ❌ Lighting dependency | ✅ Works in any lighting |
| ❌ Background noise | ✅ Compact skeletal representation |

---

## ✨ Features

- **🎥 Real-Time Capture** — MediaPipe-powered hand tracking at 30fps from any webcam
- **🦴 3D Skeleton Data** — 21 hand joints × 3 coordinates per frame
- **📦 Dataset Management** — Build labeled datasets with train/validation/test splits
- **🧠 Multiple Models** — ST-GCN, Transformer, or Hybrid architectures
- **📈 Training Dashboard** — Monitor training progress with real-time metrics
- **🌍 Multi-Language** — Support for 8+ sign languages (ASL, BdSL, BSL, ISL, JSL, CSL, DGS, LSF)
- **🔌 REST API** — Full API access for integration with external systems
- **⚡ WebSocket Streaming** — Real-time inference for live translation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (or SQLite for development)
- Redis (optional, for WebSocket support)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sumba.git
cd sumba/signlang3d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
cd backend
python manage.py migrate

# Load initial data (languages, joints, tags, model architectures)
python manage.py setup_initial_data

# Create admin user (optional)
python manage.py createsuperuser

# Start development server
USE_SQLITE=true python manage.py runserver
```

Visit **http://localhost:8000** to access the platform.

### Default Credentials

After running `setup_initial_data`:
- **Username:** `admin`
- **Password:** `admin123`

---

## 📊 Supported Languages

| Flag | Code | Language |
|------|------|----------|
| 🇺🇸 | ASL | American Sign Language |
| 🇧🇩 | BdSL | Bangladeshi Sign Language |
| 🇬🇧 | BSL | British Sign Language |
| 🇮🇳 | ISL | Indian Sign Language |
| 🇯🇵 | JSL | Japanese Sign Language |
| 🇨🇳 | CSL | Chinese Sign Language |
| 🇩🇪 | DGS | German Sign Language |
| 🇫🇷 | LSF | French Sign Language |

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Webcam        │───▶│  MediaPipe      │───▶│  ST-GCN/Trans   │───▶│  Text Decoder   │
│   (30 fps)      │    │  (21 joints×3D) │    │  (Encoder)      │    │  (Transformer)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
        ▼                      ▼                      ▼                      ▼
   Camera input          Joint positions        Motion embeddings       Text output
                        (x, y, z) × T          (B, T', D)             "Hello"
```

### Model Architectures

| Model | Description | Best For |
|-------|-------------|----------|
| **ST-GCN** | Spatial-Temporal Graph ConvNet | Fast inference, smaller datasets |
| **Transformer** | Self-attention encoder | Large datasets, complex gestures |
| **Hybrid** | ST-GCN + Transformer | Best accuracy, research use |

---

## 📁 Project Structure

```
signlang3d/
├── backend/                    # Django Backend
│   ├── core/                   # Project settings & routing
│   ├── accounts/               # User management & profiles
│   ├── gestures/               # Gesture samples & WebSocket
│   ├── datasets/               # Dataset versioning & splits
│   ├── training/               # Training runs & checkpoints
│   ├── inference/              # Inference & model deployment
│   └── api/                    # REST API (DRF)
├── ml/                         # PyTorch ML Code
│   ├── models/
│   │   ├── stgcn.py           # ST-GCN implementation
│   │   ├── motion_transformer.py
│   │   └── decoder.py         # Text decoder
│   ├── datasets/
│   │   └── sign_language.py   # PyTorch Dataset
│   ├── train.py               # Training pipeline
│   └── infer.py               # Inference pipeline
├── frontend/
│   └── templates/             # Django templates (Tailwind CSS)
├── checkpoints/               # Saved model weights
├── media/                     # Uploaded files
└── requirements.txt
```

---

## 🔌 API Reference

### REST Endpoints

```bash
# List gesture samples
GET /api/gestures/

# Create gesture sample
POST /api/gestures/
{
    "language": "ASL",
    "gloss": "hello",
    "frames": [{"joints": [[x,y,z], ...], "timestamp": 0}, ...]
}

# Run inference
POST /api/inference/
{
    "model": "hybrid",
    "language": "ASL",
    "frames": [...]
}
```

### WebSocket Endpoints

```javascript
// Connect to gesture capture
const ws = new WebSocket('ws://localhost:8000/ws/gesture/capture/');

// Send frame data
ws.send(JSON.stringify({ 
    type: 'frame', 
    data: { timestamp: 123, joints: [...] } 
}));

// Receive response
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Translation:', data.translation);
};
```

---

## 🏋️ Training

### Run Training

```bash
python ml/train.py \
    --data_dir data/ASL \
    --model_type hybrid \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --output_dir checkpoints/hybrid_asl_v1
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_type` | hybrid | Architecture (stgcn, transformer, hybrid) |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 50 | Number of epochs |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--warmup_steps` | 1000 | LR warmup steps |
| `--gradient_clip` | 1.0 | Gradient clipping |
| `--label_smoothing` | 0.1 | Label smoothing factor |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU-4** | Translation quality (higher is better) |
| **WER** | Word Error Rate (lower is better) |
| **CER** | Character Error Rate (lower is better) |

---

## 🐳 Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8000
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 📚 Citation

If you use SUMBA in your research, please cite:

```bibtex
@software{sumba2026,
    title={SUMBA: 3D Sign Language Translation Platform},
    author={Your Name},
    year={2026},
    url={https://github.com/yourusername/sumba}
}
```

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) — Hand tracking
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Django](https://www.djangoproject.com/) — Web framework
- [Tailwind CSS](https://tailwindcss.com/) — Styling

---

<p align="center">
  Made with ❤️ for the Deaf and Hard of Hearing community
</p>
