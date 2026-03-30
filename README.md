# 🎙️ Beatrice Voice Changer Training WebUI - Kaggle Version

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange.svg)](https://gradio.app/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Train custom voice models for real-time voice conversion on Kaggle's free GPU!**

---

## 📦 Các file trong package

```
beatrice-kaggle-package/
├── beatrice_webui_kaggle.py    # Main WebUI file (single-file version)
├── kaggle_notebook.py          # Quick setup script cho Kaggle
├── requirements.txt            # Dependencies
├── KAGGLE_SETUP.md            # Chi tiết setup guide
└── README.md                  # File này
```

---

## 🚀 Quick Start (3 bước)

### Bước 1: Tạo Kaggle Notebook
1. Đăng nhập [Kaggle.com](https://www.kaggle.com)
2. **Create** → **New Notebook**
3. **Settings** → **Accelerator** → Chọn **GPU T4 x2**

### Bước 2: Setup trong Kaggle Notebook

**Cell 1** - Install dependencies:
```python
!pip install -q gradio==4.44.0 torch==2.1.0 torchaudio==2.1.0 pydub pysrt
!pip install -q git+https://github.com/m-bain/whisperx.git
```

**Cell 2** - Upload và chạy setup script:
```python
# Upload file kaggle_notebook.py hoặc copy-paste toàn bộ nội dung vào cell
# Hoặc download từ URL:
!wget https://YOUR_URL/kaggle_notebook.py
!python kaggle_notebook.py
```

**Cell 3** - Upload WebUI file:
```python
# Option A: Upload beatrice_webui_kaggle.py qua Kaggle Dataset
# Option B: Download từ URL
!wget https://YOUR_URL/beatrice_webui_kaggle.py -O /kaggle/working/beatrice_webui_kaggle.py
```

### Bước 3: Chạy WebUI
```python
!python /kaggle/working/beatrice_webui_kaggle.py
```

→ Click vào link `https://xxxxx.gradio.live` để mở WebUI!

---

## 📋 Hướng dẫn chi tiết

Xem file [KAGGLE_SETUP.md](KAGGLE_SETUP.md) để có hướng dẫn đầy đủ.

---

## 🎯 Cách sử dụng WebUI

### 1️⃣ Tab "Create Dataset"
- Upload dataset (cấu trúc: `dataset/speaker1/*.wav`, `dataset/speaker2/*.wav`)
- Click **Begin Process** để transcribe audio
- Đợi quá trình split audio hoàn tất

### 2️⃣ Tab "Train"
- Chọn dataset đã process
- Điều chỉnh parameters:
  - **Batch Size:** 8-16 (tối ưu cho Kaggle GPU)
  - **Epochs:** 20-50
  - **Workers:** 2-4
- Click **Start Training**
- Monitor progress trong console

### 3️⃣ Download Model
Sau khi training xong, download model về máy:

```python
# Trong Kaggle cell
!zip -r trained_model.zip /kaggle/working/trained_models/

# Download
from IPython.display import FileLink
FileLink('trained_model.zip')
```

---

## ⚙️ Cấu hình khuyến nghị

| Kích thước Dataset | Batch Size | Epochs | Workers | Thời gian ước tính |
|-------------------|------------|--------|---------|-------------------|
| Nhỏ (< 500 files) | 8 | 50 | 2 | ~1 giờ |
| Trung bình (500-2000) | 12 | 30 | 2 | ~2-3 giờ |
| Lớn (> 2000 files) | 16 | 20 | 4 | ~4-6 giờ |

---

## 📊 Dataset Structure

```
datasets/
└── my_voice_dataset/
    ├── speaker1/          # Giọng người 1
    │   ├── audio1.wav
    │   ├── audio2.mp3
    │   └── audio3.flac
    └── speaker2/          # Giọng người 2
        ├── audio1.wav
        └── audio2.wav
```

**Yêu cầu:**
- ✅ Mỗi speaker folder tối thiểu **10 phút audio**
- ✅ Audio chất lượng tốt (16kHz+, ít noise)
- ✅ Format: WAV, MP3, FLAC, M4A, OGG
- ✅ Giọng nói rõ ràng, ít background noise

---

## 🔧 Troubleshooting

### ❌ "CUDA out of memory"
**Giải pháp:** Giảm Batch Size xuống 4 hoặc 2

### ❌ "WhisperX not found"
**Giải pháp:**
```python
!pip uninstall -y whisperx
!pip install git+https://github.com/m-bain/whisperx.git
```

### ❌ "No module named 'beatrice_trainer'"
**Giải pháp:** Install beatrice_trainer package
```python
!pip install git+https://github.com/YOUR_USERNAME/beatrice-trainer.git
```

### ❌ WebUI không load
**Giải pháp:**
```python
# Kill process cũ
!pkill -f gradio

# Restart
!python beatrice_webui_kaggle.py
```

---

## 💡 Tips & Tricks

### 1. Monitor GPU Usage
```python
!watch -n 1 nvidia-smi
```

### 2. Auto Backup (chạy trong cell riêng)
```python
import time, shutil
from datetime import datetime

while True:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    shutil.make_archive(f'backup_{timestamp}', 'zip', '/kaggle/working/trained_models')
    time.sleep(1800)  # Backup mỗi 30 phút
```

### 3. Tensorboard Monitoring
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

### 4. Tận dụng Kaggle GPU
- Sử dụng **Batch Size lớn nhất** mà GPU cho phép (thử 16 → 12 → 8)
- **Save interval = 5-10** để tránh mất data khi session timeout
- **Enable Internet** trong Settings để download models

---

## ⚠️ Giới hạn Kaggle

| Resource | Free Tier | Limit |
|----------|-----------|-------|
| GPU Time | 30 giờ/tuần | 42 giờ/tháng |
| Session | 12 giờ | Auto-disconnect |
| Disk | 20GB | Working directory |
| RAM | 13GB | GPU notebook |
| GPU Memory | 16GB | T4/P100 |

→ **Lưu model thường xuyên!**

---

## 📚 Dependencies

- **gradio** >= 4.44.0 - Web UI framework
- **torch** >= 2.1.0 - PyTorch deep learning
- **whisperx** - Speech transcription
- **pydub** - Audio processing
- **pysrt** - Subtitle handling
- **tensorboard** - Training visualization

Full list: [requirements.txt](requirements.txt)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Beatrice Trainer Team** - Original training framework
- **WhisperX** - Speech transcription
- **Gradio Team** - Amazing UI framework
- **Kaggle** - Free GPU resources

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email:** your-email@example.com

---

## 📈 Project Status

✅ **Stable** - Ready for production use on Kaggle

**Features:**
- ✅ Single-file deployment
- ✅ Auto-detect Kaggle environment
- ✅ GPU-optimized settings
- ✅ Progress tracking
- ✅ Model checkpointing
- ✅ Tensorboard integration

**Planned:**
- 🔄 Colab support
- 🔄 Model export to ONNX
- 🔄 Multi-GPU training
- 🔄 Real-time inference demo

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Made with ❤️ for the voice conversion community**

🎉 **Happy Training!**
