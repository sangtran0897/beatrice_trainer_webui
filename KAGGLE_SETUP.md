# 🚀 Beatrice Voice Training WebUI - Kaggle Setup Guide

## 📋 Hướng dẫn chạy trên Kaggle

### 🎯 Bước 1: Tạo Kaggle Notebook mới

1. Đăng nhập vào [Kaggle](https://www.kaggle.com)
2. Click **"Create"** → **"New Notebook"**
3. Đặt tên notebook: `beatrice-voice-training`

### ⚙️ Bước 2: Cấu hình Kaggle Notebook

#### 2.1. Bật GPU
- Click **Settings** (góc phải)
- Chọn **Accelerator** → **GPU T4 x2** (hoặc GPU P100)
- Click **Save**

#### 2.2. Cấu hình Persistence
- Trong **Settings**
- Bật **Internet** (để download dependencies)
- Bật **GPU Persistence** (nếu có)

### 📦 Bước 3: Upload file lên Kaggle

#### Option A: Upload trực tiếp vào Notebook

```python
# Cell 1: Upload file beatrice_webui_kaggle.py
from google.colab import files
uploaded = files.upload()
```

#### Option B: Tạo Kaggle Dataset (KHUYẾN NGHỊ)

1. Tạo dataset mới trên Kaggle:
   - Click **"Create"** → **"New Dataset"**
   - Upload `beatrice_webui_kaggle.py`
   - Đặt tên: `beatrice-webui`

2. Add dataset vào notebook:
   - Click **"+ Add data"**
   - Chọn dataset `beatrice-webui`

### 🔧 Bước 4: Cài đặt Dependencies

Copy-paste code sau vào **Cell 1** của Kaggle Notebook:

```python
# Cell 1: Install dependencies
!pip install -q gradio==4.44.0
!pip install -q git+https://github.com/m-bain/whisperx.git
!pip install -q torch==2.1.0 torchaudio==2.1.0
!pip install -q pydub pysrt
!pip install -q tensorboard

print("✅ Dependencies installed successfully!")
```

### 📁 Bước 5: Chuẩn bị Dataset

#### Option A: Upload dataset qua UI (Đơn giản nhất)

```python
# Cell 2: Create directories and upload interface
import os
from pathlib import Path

# Create directories
os.makedirs('/kaggle/working/datasets', exist_ok=True)
os.makedirs('/kaggle/working/training', exist_ok=True)
os.makedirs('/kaggle/working/trained_models', exist_ok=True)

print("✅ Directories created!")
print("\nℹ️ Upload your dataset to /kaggle/working/datasets/")
print("Structure should be:")
print("  datasets/")
print("    └── my_dataset/")
print("        ├── speaker1/")
print("        │   ├── audio1.wav")
print("        │   └── audio2.wav")
print("        └── speaker2/")
print("            ├── audio1.wav")
print("            └── audio2.wav")
```

#### Option B: Load từ Kaggle Dataset (Nếu đã upload dataset)

```python
# Cell 2: Copy dataset from Kaggle input
import shutil
from pathlib import Path

# Assuming you uploaded dataset as Kaggle Dataset
input_dataset = Path('/kaggle/input/your-dataset-name')
output_dataset = Path('/kaggle/working/datasets/my_dataset')

if input_dataset.exists():
    shutil.copytree(input_dataset, output_dataset)
    print(f"✅ Dataset copied to {output_dataset}")
else:
    print("❌ Dataset not found. Please upload via Kaggle Dataset.")
```

#### Option C: Download từ internet

```python
# Cell 2: Download dataset from URL
!wget -O dataset.zip "YOUR_DATASET_URL"
!unzip -q dataset.zip -d /kaggle/working/datasets/
!rm dataset.zip

print("✅ Dataset downloaded and extracted!")
```

### 🚀 Bước 6: Chạy WebUI

```python
# Cell 3: Run Beatrice WebUI
!python /kaggle/input/beatrice-webui/beatrice_webui_kaggle.py

# Hoặc nếu upload trực tiếp:
# !python beatrice_webui_kaggle.py
```

**Sau khi chạy, Gradio sẽ tạo public link:**
```
Running on public URL: https://xxxxx.gradio.live
```

→ Click vào link để mở WebUI!

### 📊 Bước 7: Sử dụng WebUI

#### 7.1. Tab "Create Dataset"
1. Chọn dataset từ dropdown
2. Click **"Begin Process"** để transcribe và split audio
3. Đợi quá trình hoàn tất

#### 7.2. Tab "Train"
1. Chọn dataset đã process
2. Điều chỉnh parameters:
   - **Batch Size:** 8-16 (cho Kaggle GPU)
   - **Epochs:** 20-50
   - **Workers:** 2-4
3. Click **"Start Training"**
4. Monitor training qua console

#### 7.3. Download Model
Sau khi training xong:

```python
# Cell 4: Zip and download trained model
!zip -r trained_model.zip /kaggle/working/trained_models/

from IPython.display import FileLink
FileLink('trained_model.zip')
```

### 📈 Bước 8: Monitor Training với Tensorboard (Optional)

```python
# Cell 5: Launch Tensorboard
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

---

## 🔥 QUICK START - Copy toàn bộ vào 1 cell

```python
# QUICK START - All-in-one setup
# ================================

# 1. Install dependencies
!pip install -q gradio==4.44.0
!pip install -q git+https://github.com/m-bain/whisperx.git
!pip install -q torch==2.1.0 torchaudio==2.1.0
!pip install -q pydub pysrt tensorboard

# 2. Create directories
import os
os.makedirs('/kaggle/working/datasets', exist_ok=True)
os.makedirs('/kaggle/working/training', exist_ok=True)
os.makedirs('/kaggle/working/trained_models', exist_ok=True)

# 3. Copy WebUI file (if uploaded as dataset)
!cp /kaggle/input/beatrice-webui/beatrice_webui_kaggle.py /kaggle/working/

# 4. Launch WebUI
!python /kaggle/working/beatrice_webui_kaggle.py
```

---

## ⚠️ Lưu ý quan trọng

### 1. Thời gian Kaggle Session
- **Kaggle Free:** 9 giờ GPU/tuần
- **Kaggle Pro:** 30 giờ GPU/tuần
- Session timeout: 12 giờ

### 2. Lưu Model trước khi hết session
```python
# Định kỳ save model về local
!zip -r model_checkpoint.zip /kaggle/working/trained_models/
from IPython.display import FileLink
FileLink('model_checkpoint.zip')
```

### 3. Giới hạn Kaggle
- **Disk space:** 20GB working directory
- **RAM:** 13GB (GPU notebook)
- **GPU Memory:** 16GB (T4) hoặc 16GB (P100)

### 4. Tối ưu cho Kaggle
- Sử dụng **batch size lớn** (8-16) để tận dụng GPU
- **Workers = 2** (Kaggle CPU không quá mạnh)
- **Save interval = 5-10** epochs để không mất tiến độ

---

## 🐛 Troubleshooting

### Lỗi: "CUDA out of memory"
```python
# Giảm batch size
# Trong WebUI: Batch Size = 4 hoặc 2
```

### Lỗi: "WhisperX not found"
```python
# Reinstall WhisperX
!pip uninstall -y whisperx
!pip install git+https://github.com/m-bain/whisperx.git
```

### Lỗi: "No module named 'beatrice_trainer'"
```python
# Install beatrice_trainer
!pip install git+https://github.com/yourusername/beatrice-trainer.git

# Hoặc nếu có local:
!pip install -e /kaggle/input/beatrice-trainer/
```

### WebUI không load
```python
# Check if Gradio is running
!ps aux | grep python

# Kill old process
!pkill -f gradio

# Restart
!python beatrice_webui_kaggle.py
```

---

## 📚 Tài liệu tham khảo

- [Kaggle GPU Quota](https://www.kaggle.com/discussions/product-feedback/280513)
- [WhisperX GitHub](https://github.com/m-bain/whisperx)
- [Gradio Documentation](https://gradio.app/docs/)

---

## 💡 Tips & Tricks

### 1. Tận dụng tối đa GPU
```python
# Trong WebUI settings:
# - Batch Size: 16 (nếu GPU T4/P100)
# - Num Workers: 2
# - Use mixed precision training (if supported)
```

### 2. Monitor GPU usage
```python
# Cell riêng để monitor GPU
!watch -n 1 nvidia-smi
```

### 3. Backup định kỳ
```python
# Auto backup mỗi 30 phút
import time
import shutil
from datetime import datetime

while True:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.make_archive(f'backup_{timestamp}', 'zip', '/kaggle/working/trained_models')
    time.sleep(1800)  # 30 minutes
```

### 4. Sử dụng Kaggle Datasets để chia sẻ model
```python
# Upload trained model as Kaggle Dataset để dùng cho lần sau
# File -> Save Version -> Save & Run All -> Quick Save
```

---

## ✅ Checklist

- [ ] Tạo Kaggle Notebook
- [ ] Bật GPU (T4 hoặc P100)
- [ ] Upload file `beatrice_webui_kaggle.py`
- [ ] Install dependencies
- [ ] Upload dataset
- [ ] Chạy WebUI
- [ ] Process dataset
- [ ] Configure training parameters
- [ ] Start training
- [ ] Monitor progress
- [ ] Download trained model

---

**Happy Training! 🎉**
