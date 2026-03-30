#!/usr/bin/env python3
"""
Beatrice Training WebUI - Kaggle Notebook Script
=================================================
Copy toàn bộ script này vào 1 cell duy nhất trong Kaggle Notebook
"""

# ==============================================================================
# CELL 1: SETUP & INSTALL
# ==============================================================================

print("="*80)
print("🚀 BEATRICE VOICE TRAINING WEBUI - KAGGLE SETUP")
print("="*80)

# Install dependencies
print("\n📦 Installing dependencies...")
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Core packages
packages = [
    "gradio==4.44.0",
    "torch==2.1.0",
    "torchaudio==2.1.0",
    "pydub",
    "pysrt",
    "tensorboard"
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    install(pkg)

# WhisperX from git
print("  Installing WhisperX from git...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/m-bain/whisperx.git"
])

print("✅ All dependencies installed!\n")

# ==============================================================================
# CELL 2: ENVIRONMENT CHECK
# ==============================================================================

import torch
import os
from pathlib import Path

print("="*80)
print("🔍 ENVIRONMENT CHECK")
print("="*80)

print(f"\n✅ Python: {sys.version}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ WARNING: No GPU detected! Please enable GPU in Kaggle settings.")
    print("   Settings → Accelerator → GPU T4 x2")

# ==============================================================================
# CELL 3: SETUP DIRECTORIES
# ==============================================================================

print("\n" + "="*80)
print("📁 SETTING UP DIRECTORIES")
print("="*80 + "\n")

# Create necessary directories
base_dir = Path("/kaggle/working")
directories = {
    "datasets": base_dir / "datasets",
    "training": base_dir / "training",
    "trained_models": base_dir / "trained_models",
    "configs": base_dir / "configs",
    "logs": base_dir / "logs",
    "assets": base_dir / "assets"
}

for name, path in directories.items():
    path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {path}")

print("\n📊 Directory structure:")
print("  /kaggle/working/")
print("  ├── datasets/         # Upload your datasets here")
print("  ├── training/         # Processed training data")
print("  ├── trained_models/   # Output models")
print("  ├── configs/          # Configuration files")
print("  ├── logs/             # Training logs")
print("  └── assets/           # Assets")

# ==============================================================================
# CELL 4: DOWNLOAD WEBUI SCRIPT
# ==============================================================================

print("\n" + "="*80)
print("📥 DOWNLOADING WEBUI SCRIPT")
print("="*80 + "\n")

# Check if script exists
webui_path = base_dir / "beatrice_webui_kaggle.py"

if not webui_path.exists():
    print("⚠️ WebUI script not found!")
    print("\nPlease do ONE of the following:")
    print("\n1. Upload via Kaggle Dataset:")
    print("   - Create new dataset with beatrice_webui_kaggle.py")
    print("   - Add dataset to this notebook")
    print("   - Then run: !cp /kaggle/input/YOUR-DATASET/beatrice_webui_kaggle.py /kaggle/working/")
    print("\n2. Upload directly:")
    print("   - Use Kaggle's 'Add Data' → 'Upload'")
    print("   - Upload beatrice_webui_kaggle.py")
    print("\n3. Download from URL (if hosted online):")
    print("   !wget -O /kaggle/working/beatrice_webui_kaggle.py YOUR_URL")
else:
    print(f"✅ WebUI script found: {webui_path}")

# ==============================================================================
# CELL 5: EXAMPLE DATASET SETUP
# ==============================================================================

print("\n" + "="*80)
print("📂 DATASET SETUP INSTRUCTIONS")
print("="*80 + "\n")

print("Your dataset should follow this structure:\n")
print("  datasets/")
print("  └── my_voice_dataset/")
print("      ├── speaker1/")
print("      │   ├── audio1.wav")
print("      │   ├── audio2.wav")
print("      │   └── audio3.wav")
print("      └── speaker2/")
print("          ├── audio1.wav")
print("          └── audio2.wav")

print("\n💡 How to upload dataset:\n")
print("Option A: Via Kaggle Dataset (RECOMMENDED)")
print("  1. Create new Kaggle Dataset")
print("  2. Upload your dataset folder")
print("  3. Add dataset to this notebook")
print("  4. Copy to working directory:")
print("     !cp -r /kaggle/input/YOUR-DATASET/* /kaggle/working/datasets/")

print("\nOption B: Direct upload")
print("  1. Create empty folder: !mkdir -p /kaggle/working/datasets/my_dataset/speaker1")
print("  2. Upload files via Kaggle UI")

print("\nOption C: Download from URL")
print("  !wget -O dataset.zip YOUR_URL")
print("  !unzip dataset.zip -d /kaggle/working/datasets/")

# ==============================================================================
# CELL 6: LAUNCH WEBUI
# ==============================================================================

print("\n" + "="*80)
print("🚀 READY TO LAUNCH WEBUI")
print("="*80 + "\n")

if webui_path.exists():
    print("Run the following command in the NEXT cell:\n")
    print("!python /kaggle/working/beatrice_webui_kaggle.py")
    print("\nThis will:")
    print("  1. Start Gradio WebUI")
    print("  2. Generate a public link (https://xxxxx.gradio.live)")
    print("  3. Open the link to access the UI")
    print("\n⚠️ NOTE: The link expires when the notebook stops running!")
else:
    print("❌ WebUI script not found. Please complete CELL 4 first.")

# ==============================================================================
# CELL 7: HELPER FUNCTIONS
# ==============================================================================

print("\n" + "="*80)
print("🔧 HELPER FUNCTIONS LOADED")
print("="*80 + "\n")

def check_gpu_usage():
    """Check GPU memory usage"""
    os.system("nvidia-smi")

def backup_models(output_name="trained_models_backup"):
    """Backup trained models"""
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_name}_{timestamp}"
    
    shutil.make_archive(
        output_file,
        'zip',
        '/kaggle/working/trained_models'
    )
    
    print(f"✅ Backup created: {output_file}.zip")
    print(f"   Size: {os.path.getsize(f'{output_file}.zip') / 1024**2:.2f} MB")
    
    return f"{output_file}.zip"

def download_file(filename):
    """Create download link for file"""
    from IPython.display import FileLink
    return FileLink(filename)

def list_datasets():
    """List available datasets"""
    datasets_dir = Path("/kaggle/working/datasets")
    
    if not datasets_dir.exists():
        print("❌ No datasets directory found")
        return
    
    datasets = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    if not datasets:
        print("📂 No datasets found. Please upload your dataset.")
        return
    
    print(f"📂 Found {len(datasets)} dataset(s):\n")
    for dataset in datasets:
        speakers = [s for s in dataset.iterdir() if s.is_dir()]
        total_files = sum(
            1 for speaker in speakers 
            for f in speaker.iterdir() 
            if f.is_file()
        )
        print(f"  • {dataset.name}")
        print(f"    - Speakers: {len(speakers)}")
        print(f"    - Audio files: {total_files}")

def list_trained_models():
    """List trained models"""
    models_dir = Path("/kaggle/working/trained_models")
    
    if not models_dir.exists():
        print("❌ No trained models directory found")
        return
    
    models = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not models:
        print("📦 No trained models yet. Start training first!")
        return
    
    print(f"📦 Found {len(models)} trained model(s):\n")
    for model in models:
        model_files = list((model / "models").glob("*.pt")) if (model / "models").exists() else []
        size = sum(f.stat().st_size for f in model_files) / 1024**2
        
        print(f"  • {model.name}")
        print(f"    - Checkpoints: {len(model_files)}")
        print(f"    - Size: {size:.2f} MB")

# Test helper functions
print("Helper functions available:")
print("  • check_gpu_usage()       - Monitor GPU memory")
print("  • backup_models()         - Backup trained models")
print("  • download_file(path)     - Create download link")
print("  • list_datasets()         - List uploaded datasets")
print("  • list_trained_models()   - List trained models")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("✅ SETUP COMPLETE!")
print("="*80 + "\n")

print("📋 Next steps:\n")
print("1. Upload your dataset to /kaggle/working/datasets/")
print("   Run: list_datasets() to verify")
print("\n2. Launch WebUI:")
print("   !python /kaggle/working/beatrice_webui_kaggle.py")
print("\n3. In WebUI:")
print("   - Tab 'Create Dataset': Process your audio")
print("   - Tab 'Train': Configure and start training")
print("\n4. Download trained model:")
print("   backup_models()")
print("   download_file('trained_models_backup_*.zip')")

print("\n💡 Tips:")
print("  • Monitor GPU: check_gpu_usage()")
print("  • Training takes 1-3 hours for ~1000 files")
print("  • Save frequently (Kaggle session = 12 hours max)")
print("  • Batch size 8-16 optimal for Kaggle GPU")

print("\n🎉 Happy Training!")
print("="*80)
