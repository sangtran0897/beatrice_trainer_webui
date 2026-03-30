#!/usr/bin/env python3
"""
Beatrice Voice Changer Training WebUI - Kaggle Optimized Version
==================================================================
Single-file version optimized for Kaggle environment

Author: Beatrice Team (Modified for Kaggle)
License: MIT
"""

import os
import sys
import shutil
import json
import math
import time
import subprocess
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional

import gradio as gr
import tqdm
import pysrt
from pydub import AudioSegment

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

IN_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')
IN_COLAB = 'COLAB_GPU' in os.environ

print("=" * 80)
if IN_KAGGLE:
    print("🚀 Running in KAGGLE environment")
elif IN_COLAB:
    print("🚀 Running in COLAB environment")
else:
    print("🖥️  Running in LOCAL environment")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for different environments"""
    
    # Base directories
    if IN_KAGGLE:
        BASE_DIR = Path("/kaggle/working")
        INPUT_DIR = Path("/kaggle/input")
    else:
        BASE_DIR = Path.cwd()
        INPUT_DIR = BASE_DIR / "inputs"
    
    # Working directories
    DATASETS_DIR = BASE_DIR / "datasets"
    TRAINING_DIR = BASE_DIR / "training"
    TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
    CONFIGS_DIR = BASE_DIR / "configs"
    LOGS_DIR = BASE_DIR / "logs"
    ASSETS_DIR = BASE_DIR / "assets"
    
    # Valid audio extensions
    VALID_AUDIO_EXT = [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".opus", ".mp4"]
    
    # Training defaults (optimized for Kaggle GPU)
    DEFAULT_BATCH_SIZE = 8 if (IN_KAGGLE or IN_COLAB) else 4
    DEFAULT_EPOCHS = 20
    DEFAULT_NUM_WORKERS = 2 if (IN_KAGGLE or IN_COLAB) else 4
    DEFAULT_SAVE_INTERVAL = 5
    DEFAULT_LOG_INTERVAL = 50
    
    # Gradio settings
    GRADIO_SERVER_NAME = "0.0.0.0" if (IN_KAGGLE or IN_COLAB) else "127.0.0.1"
    GRADIO_SERVER_PORT = 7860
    GRADIO_SHARE = IN_KAGGLE or IN_COLAB
    GRADIO_INBROWSER = not (IN_KAGGLE or IN_COLAB)
    
    @classmethod
    def setup_directories(cls):
        """Create all necessary directories"""
        for dir_path in [
            cls.DATASETS_DIR,
            cls.TRAINING_DIR,
            cls.TRAINED_MODELS_DIR,
            cls.CONFIGS_DIR,
            cls.LOGS_DIR,
            cls.ASSETS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directories in: {cls.BASE_DIR}")

# Setup directories on import
Config.setup_directories()

# ============================================================================
# UTILITY FUNCTIONS (Inline to avoid dependencies)
# ============================================================================

def get_available_items(root: str, directory_only: bool = False, 
                        extensions: Optional[List[str]] = None) -> List[str]:
    """
    Get available items in a directory
    
    Args:
        root: Root directory path
        directory_only: If True, only return directories
        extensions: List of file extensions to filter (e.g., ['.wav', '.mp3'])
    
    Returns:
        List of available items
    """
    root_path = Path(root)
    
    if not root_path.exists():
        return []
    
    items = []
    
    for item in root_path.iterdir():
        if directory_only:
            if item.is_dir():
                items.append(str(item))
        else:
            if extensions:
                if item.suffix.lower() in extensions:
                    items.append(str(item))
            else:
                items.append(str(item))
    
    return sorted(items)


def refresh_dropdown_proxy(root: str, extensions_json: str, option: str):
    """
    Refresh dropdown choices
    
    Args:
        root: Root directory
        extensions_json: JSON string of extensions
        option: 'directory' or 'files'
    
    Returns:
        Updated Dropdown component
    """
    import json
    
    extensions = json.loads(extensions_json) if extensions_json else None
    directory_only = (option == "directory")
    
    items = get_available_items(root, directory_only, extensions)
    
    return gr.Dropdown(choices=items)


def move_existing_folder(source_root: str, folder_name: str, destination_root: str):
    """Move folder from source to destination"""
    source_path = Path(source_root) / folder_name
    destination_path = Path(destination_root)
    
    if not source_path.exists():
        gr.Warning(f"Source folder not found: {source_path}")
        return
    
    destination_path.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(str(source_path), str(destination_path))
        gr.Info(f"✅ Moved {folder_name} to {destination_root}")
    except Exception as e:
        gr.Error(f"❌ Failed to move folder: {e}")


def launch_tensorboard_proxy():
    """Launch Tensorboard"""
    try:
        log_dir = Config.LOGS_DIR
        
        if not log_dir.exists():
            gr.Warning("No logs directory found. Train a model first.")
            return
        
        # Kill existing tensorboard
        subprocess.run(["pkill", "-f", "tensorboard"], check=False)
        
        # Launch new tensorboard
        cmd = f"tensorboard --logdir {log_dir} --port 6006 --bind_all"
        subprocess.Popen(cmd, shell=True)
        
        if IN_KAGGLE or IN_COLAB:
            gr.Info("✅ Tensorboard started at port 6006")
        else:
            gr.Info("✅ Tensorboard started at http://localhost:6006")
    except Exception as e:
        gr.Error(f"❌ Failed to launch Tensorboard: {e}")

# ============================================================================
# DATASET PROCESSING FUNCTIONS
# ============================================================================

def is_correct_dataset_structure(folder_to_analyze: str) -> bool:
    """
    Check if dataset folder has correct structure (only contains folders)
    """
    folder_path = Path(folder_to_analyze)
    
    if not folder_path.exists() or not folder_path.is_dir():
        return False
    
    items = list(folder_path.iterdir())
    
    if len(items) == 0:
        return False
    
    for item in items:
        if not item.is_dir():
            return False
    
    return True


def folder_to_process_proxy(folder_to_analyze: str):
    """Validate folder structure before processing"""
    folder_check = is_correct_dataset_structure(folder_to_analyze)
    
    if not folder_check:
        raise gr.Error(
            "❌ Invalid folder structure!\n"
            "Please ensure:\n"
            "1. Folder is NOT empty\n"
            "2. Folder contains ONLY subfolders (one per speaker)\n"
            "3. Each subfolder contains audio files"
        )
    
    return gr.Dropdown(value=folder_to_analyze)


def load_whisperx(model_name: str = 'large-v3', progress=None):
    """
    Load WhisperX model
    
    Args:
        model_name: Whisper model name
        progress: Gradio progress tracker
    
    Returns:
        Loaded WhisperX model
    """
    import whisperx
    import torch
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        raise gr.Error("❌ CUDA not available! This tool requires a GPU.")
    
    try:
        print(f"Loading WhisperX model: {model_name}...")
        whisper_model = whisperx.load_model(
            model_name, 
            device, 
            download_root="whisper_models", 
            compute_type="float16"
        )
        print("✅ Loaded Whisper model with float16")
    except Exception as e:
        print(f"⚠️ Float16 failed, trying int8: {e}")
        whisper_model = whisperx.load_model(
            model_name, 
            device, 
            download_root="whisper_models", 
            compute_type="int8"
        )
        print("✅ Loaded Whisper model with int8")
    
    return whisper_model


def run_whisperx_transcribe(audio_file_path: str, whisper_model, 
                            chunk_size: int = 15, language: Optional[str] = None):
    """
    Transcribe audio file using WhisperX
    
    Args:
        audio_file_path: Path to audio file
        whisper_model: Loaded WhisperX model
        chunk_size: Chunk size for processing
        language: Language code (auto-detect if None)
    
    Returns:
        Transcription result with alignment
    """
    import whisperx
    
    audio = whisperx.load_audio(audio_file_path)
    result = whisper_model.transcribe(audio=audio, batch_size=16, chunk_size=chunk_size)
    
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device="cuda"
    )
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device="cuda", 
        return_char_alignments=False
    )
    
    # Add language tag if missing
    if "language" not in result:
        result["language"] = language or result.get("language", "en")
    
    return result


def run_whisperx_srt(transcription_result, output_directory: str):
    """
    Save transcription as SRT file
    
    Args:
        transcription_result: WhisperX transcription result
        output_directory: Output directory path
    """
    from whisperx.utils import get_writer
    
    srt_writer = get_writer("srt", output_directory)
    srt_writer(
        transcription_result, 
        output_directory, 
        {
            "max_line_width": None, 
            "max_line_count": None, 
            "highlight_words": False
        }
    )


def process_speaker_folder(file_info: Tuple[str, str, str], progress_bar=None):
    """
    Process a single speaker folder - split audio by SRT timestamps
    
    Args:
        file_info: Tuple of (folder_path, audio_file, srt_file)
        progress_bar: Optional progress bar
    """
    folder_path, audio_file, srt_file = file_info
    
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)  # Convert to mono
    subs = pysrt.open(srt_file)
    
    base_name = os.path.basename(folder_path)
    segment_counter = 1
    
    max_segment_duration = 8000  # 8 seconds max
    
    for idx, sub in enumerate(tqdm.tqdm(subs, desc="Processing Subtitles", leave=False, file=sys.stdout)):
        # Convert SRT time to milliseconds
        start_time = (
            sub.start.hours * 3600 + 
            sub.start.minutes * 60 + 
            sub.start.seconds
        ) * 1000 + sub.start.milliseconds
        
        end_time = (
            sub.end.hours * 3600 + 
            sub.end.minutes * 60 + 
            sub.end.seconds
        ) * 1000 + sub.end.milliseconds
        
        duration = end_time - start_time
        
        # Split long segments
        while duration > max_segment_duration:
            segment_end_time = start_time + max_segment_duration
            segment = audio[start_time:segment_end_time]
            output_file = f"{folder_path}/{base_name}_{segment_counter}.wav"
            segment.export(output_file, format="wav")
            start_time = segment_end_time
            duration = end_time - start_time
            segment_counter += 1
        
        # Export remaining segment
        if duration > 0:
            segment = audio[start_time:end_time]
            output_file = f"{folder_path}/{base_name}_{segment_counter}.wav"
            segment.export(output_file, format="wav")
            segment_counter += 1
        
        if progress_bar:
            progress_bar.update(1)
    
    # Clean up original files
    os.remove(audio_file)
    os.remove(srt_file)


def split_by_srt(folder_path: str, progress_bar=None):
    """
    Split all audio files in folder by their corresponding SRT files
    
    Args:
        folder_path: Path to folder containing audio and SRT files
        progress_bar: Optional progress bar
    """
    file_pairs = []
    
    for file in os.listdir(folder_path):
        if file.endswith(tuple(Config.VALID_AUDIO_EXT)):
            audio_file = os.path.join(folder_path, file)
            srt_file = os.path.join(folder_path, file.rsplit('.', 1)[0] + '.srt')
            
            if os.path.exists(srt_file):
                file_pairs.append((folder_path, audio_file, srt_file))
    
    if len(file_pairs) == 0:
        print(f"⚠️ No audio-SRT pairs found in {folder_path}")
        return
    
    with Pool(cpu_count()) as pool:
        list(tqdm.tqdm(
            pool.imap_unordered(process_speaker_folder, file_pairs), 
            total=len(file_pairs), 
            desc="Processing Files", 
            file=sys.stdout
        ))


def process_proxy(folder_to_process_path: str, progress=gr.Progress(track_tqdm=True)):
    """
    Main dataset processing function
    
    Steps:
    1. Copy dataset to training directory
    2. Transcribe all audio files with WhisperX
    3. Split audio files by SRT timestamps
    
    Args:
        folder_to_process_path: Path to dataset folder
        progress: Gradio progress tracker
    
    Returns:
        Success message
    """
    import whisperx
    import torch
    
    training_destination = Config.TRAINING_DIR / Path(folder_to_process_path).name
    
    # Check if training folder already exists
    if training_destination.exists():
        raise gr.Error(
            f"❌ Training folder already exists: {training_destination}\n"
            "Please remove it or use 'Move Existing Folder' button first."
        )
    
    # Validate dataset structure
    if not is_correct_dataset_structure(folder_to_process_path):
        raise gr.Error(
            "❌ Invalid folder structure!\n"
            "Ensure the folder contains ONLY subfolders (one per speaker)."
        )
    
    try:
        # Create training directory
        training_destination.mkdir(parents=True, exist_ok=False)
        
        # Load WhisperX model
        progress(0, desc="Loading WhisperX model...")
        whisper_model = load_whisperx('large-v3')
        
        # Get list of speaker folders
        speaker_folders_list = [
            os.path.join(folder_to_process_path, folder) 
            for folder in os.listdir(folder_to_process_path)
        ]
        
        total_speakers = len(speaker_folders_list)
        
        # Process each speaker folder
        for idx, speaker_folder_path in enumerate(speaker_folders_list):
            speaker_name = os.path.basename(speaker_folder_path)
            progress(
                idx / total_speakers, 
                desc=f"Processing speaker {idx+1}/{total_speakers}: {speaker_name}"
            )
            
            speaker_folder_dest = training_destination / speaker_name
            speaker_folder_dest.mkdir(parents=True, exist_ok=False)
            
            # Get audio files in speaker folder
            audio_files = [
                f for f in os.listdir(speaker_folder_path)
                if Path(f).suffix.lower() in Config.VALID_AUDIO_EXT
            ]
            
            # Process each audio file
            for file in tqdm.tqdm(audio_files, desc=f"Processing {speaker_name}", file=sys.stdout, leave=False):
                file_path = os.path.join(speaker_folder_path, file)
                copied_path = speaker_folder_dest / file
                
                # Copy audio file
                shutil.copy(file_path, copied_path)
                
                # Transcribe
                transcription_result = run_whisperx_transcribe(str(copied_path), whisper_model)
                run_whisperx_srt(transcription_result, str(speaker_folder_dest))
                
                # Rename SRT file to match audio file
                file_name = os.path.splitext(file)[0]
                srt_orig_path = speaker_folder_dest / f"{speaker_name}.srt"
                srt_new_path = speaker_folder_dest / f"{file_name}.srt"
                
                if srt_orig_path.exists():
                    os.rename(srt_orig_path, srt_new_path)
        
        # Split all audio files by SRT timestamps
        progress(0.8, desc="Splitting audio by timestamps...")
        for folder in tqdm.tqdm(os.listdir(training_destination), desc="Splitting by SRT", file=sys.stdout):
            folder_path = training_destination / folder
            split_by_srt(str(folder_path), progress_bar=progress)
        
        progress(1.0, desc="Completed!")
        
        return f"✅ Dataset creation completed successfully!\n\nOutput: {training_destination}"
        
    except Exception as e:
        # Clean up on error
        if training_destination.exists():
            shutil.rmtree(training_destination)
        raise gr.Error(f"❌ Dataset processing failed: {str(e)}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def count_items_in_directory(root: Path) -> int:
    """Count total files in directory recursively"""
    file_count = 0
    
    for item in root.rglob('*'):
        if item.is_file():
            file_count += 1
    
    return file_count


def find_largest_folder(directory: str) -> int:
    """Find the folder with most files"""
    max_file_count = 0
    
    for root, dirs, files in os.walk(directory):
        if files:
            file_count = len(files)
            if file_count > max_file_count:
                max_file_count = file_count
    
    return max_file_count


def training_calculations(total_audio_files: int, batch_size: int, epochs: int) -> Tuple[int, int]:
    """
    Calculate training parameters
    
    Args:
        total_audio_files: Total number of audio files
        batch_size: Batch size
        epochs: Number of epochs
    
    Returns:
        (batches_per_epoch, total_steps)
    """
    batches_per_epoch = total_audio_files // batch_size
    n_steps = epochs * batches_per_epoch
    return batches_per_epoch, n_steps


def recommendation_proxy(data_dir: str, epochs: int) -> str:
    """
    Provide training recommendations based on dataset size
    
    Args:
        data_dir: Dataset directory
        epochs: Number of epochs
    
    Returns:
        Recommendation message
    """
    if not data_dir or not os.path.exists(data_dir):
        return "⚠️ Please select a valid dataset first."
    
    recommended_exposure = 10000
    largest_file_count = find_largest_folder(data_dir)
    
    if largest_file_count == 0:
        return "⚠️ No files found in dataset."
    
    user_value = largest_file_count * epochs
    recommended_epochs = math.ceil(recommended_exposure / largest_file_count)
    
    if user_value < recommended_exposure:
        return (
            f"📊 **Training Analysis:**\n\n"
            f"• Largest speaker folder: **{largest_file_count} files**\n"
            f"• Current epochs: **{epochs}**\n"
            f"• Model exposure: **{user_value:,} times**\n\n"
            f"💡 **Recommendation:**\n"
            f"Recommended exposure is **{recommended_exposure:,} times**\n"
            f"Suggested epochs: **{recommended_epochs}**\n\n"
            f"⚠️ You may find it not necessary to train longer, but that's your decision."
        )
    else:
        return (
            f"✅ **Training parameters look good!**\n\n"
            f"• Largest speaker folder: **{largest_file_count} files**\n"
            f"• Current epochs: **{epochs}**\n"
            f"• Model exposure: **{user_value:,} times**\n\n"
            f"If the model quality is lacking after training:\n"
            f"1. Add more audio files to your dataset\n"
            f"2. Train for more epochs"
        )


def training_proxy(
    data_dir: str,
    batch_size: int,
    epochs: int,
    num_workers: int,
    resume: bool,
    save_interval: int,
    log_interval: int,
    progress=gr.Progress(track_tqdm=True)
) -> str:
    """
    Main training function
    
    Args:
        data_dir: Dataset directory path
        batch_size: Batch size for training
        epochs: Number of epochs
        num_workers: Number of data loading workers
        resume: Whether to resume from checkpoint
        save_interval: Save model every N epochs
        log_interval: Log metrics every N steps
        progress: Gradio progress tracker
    
    Returns:
        Training completion message
    """
    from beatrice_trainer.src.train import run_training
    
    data_path = Path(data_dir)
    output_name = data_path.name
    output_dir = Config.TRAINED_MODELS_DIR / output_name
    models_output_dir = output_dir / "models"
    
    # Count total audio files
    progress(0, desc="Counting audio files...")
    total_audio_files = count_items_in_directory(data_path)
    
    if total_audio_files == 0:
        raise gr.Error(f"❌ No audio files found in {data_dir}")
    
    # Calculate training parameters
    progress(0.1, desc="Calculating training parameters...")
    batches_per_epoch, n_steps = training_calculations(total_audio_files, batch_size, epochs)
    
    # Calculate warmup steps (50% of total steps, max 10k)
    warmup_steps = min(n_steps // 2, 10000)
    
    print(f"\n{'='*60}")
    print(f"📊 TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Dataset: {output_name}")
    print(f"Total audio files: {total_audio_files:,}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Batches per epoch: {batches_per_epoch:,}")
    print(f"Total steps: {n_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")
    print(f"Save interval: {save_interval} epochs")
    print(f"Log interval: {log_interval} steps")
    print(f"{'='*60}\n")
    
    # Load base configuration
    config_path = Config.ASSETS_DIR / 'default_config.json'
    
    if not config_path.exists():
        # Create default config if not exists
        default_config = {
            "batch_size": batch_size,
            "n_steps": n_steps,
            "num_workers": num_workers,
            "warmup_steps": warmup_steps,
            "learning_rate": 1e-4,
            "weight_decay": 0.01
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Update configuration
    progress(0.2, desc="Preparing configuration...")
    config.update({
        'batch_size': batch_size,
        'n_steps': n_steps,
        'num_workers': num_workers,
        'warmup_steps': warmup_steps
    })
    
    # Save updated configuration
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_config_path = output_dir / 'config.json'
    
    with open(updated_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Configuration saved to: {updated_config_path}\n")
    
    # Start training
    try:
        progress(0.3, desc="Starting training...")
        run_training(
            data_path,
            models_output_dir,
            batches_per_epoch,
            save_interval,
            log_interval,
            resume,
            updated_config_path
        )
        
        return (
            f"✅ **Training completed successfully!**\n\n"
            f"Model saved to: `{models_output_dir}`\n"
            f"Configuration: `{updated_config_path}`"
        )
        
    except Exception as e:
        raise gr.Error(f"❌ Training failed: {str(e)}")

# ============================================================================
# GRADIO UI
# ============================================================================

def load_settings() -> Dict:
    """Load UI settings from file"""
    settings_file = Config.CONFIGS_DIR / 'settings.json'
    
    if not settings_file.exists():
        settings = {"custom_theme": True, "dark_mode": True}
        save_settings(settings)
    else:
        with open(settings_file, 'r') as f:
            settings = json.load(f)
    
    return settings


def save_settings(settings: Dict):
    """Save UI settings to file"""
    settings_file = Config.CONFIGS_DIR / 'settings.json'
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)


def toggle_theme():
    """Toggle between custom and default theme"""
    settings = load_settings()
    settings["custom_theme"] = not settings.get("custom_theme", False)
    save_settings(settings)
    
    if settings['custom_theme']:
        gr.Info("✅ Custom theme will be applied on next restart.")
    else:
        gr.Info("✅ Default theme will be applied on next restart.")


def toggle_dark_mode():
    """Toggle dark mode"""
    settings = load_settings()
    settings["dark_mode"] = not settings.get("dark_mode", True)
    save_settings(settings)
    
    if settings['dark_mode']:
        gr.Info("✅ Dark mode will be applied on next restart.")
    else:
        gr.Info("✅ Light mode will be applied on next restart.")


def create_ui():
    """Create Gradio UI"""
    
    # Load settings
    settings = load_settings()
    
    # Create theme
    if settings.get("custom_theme", True):
        theme = gr.themes.Glass(
            primary_hue="zinc",
            secondary_hue="slate",
            neutral_hue="orange",
            text_size="lg"
        ).set(
            body_background_fill_dark='*primary_900',
            body_text_color='*primary_950',
            body_text_color_subdued='*neutral_500',
            embed_radius='*radius_md',
            border_color_accent_subdued_dark='*neutral_950',
            border_color_primary_dark='*secondary_800',
            color_accent_soft='*primary_400',
            block_border_width_dark='0',
            block_label_border_width_dark='None',
            block_shadow_dark='*primary_600 0px 0px 5px 0px',
            button_border_width='2px',
            button_border_width_dark='0px',
            button_shadow_hover='*block_shadow',
            button_large_radius='*radius_md',
            button_small_radius='*radius_md',
            button_small_text_weight='500',
            button_primary_border_color='*primary_500',
            button_primary_border_color_dark='*primary_950'
        )
    else:
        theme = gr.themes.Default()
    
    # JavaScript for dark mode
    js_dark_mode = (
        "document.querySelector('body').classList.add('dark');" 
        if settings.get("dark_mode", True) 
        else "document.querySelector('body').classList.remove('dark');"
    )
    
    js = f"""
        function createGradioAnimation() {{
            var container = document.createElement('div');
            container.id = 'gradio-animation';
            container.style.fontSize = '2em';
            container.style.fontWeight = 'bold';
            container.style.textAlign = 'center';
            container.style.marginBottom = '20px';
            container.style.position = 'absolute';
            container.style.left = '-100%';
            container.style.top = '20px';
            container.style.transition = 'left 1s ease-out';
            container.style.zIndex = '1000';
            container.style.whiteSpace = 'nowrap';
            container.style.overflow = 'hidden';
            container.style.textOverflow = 'ellipsis';

            var text = 'Beatrice Voice Changer Training WebUI';
            container.innerText = text;

            var gradioContainer = document.querySelector('.gradio-container');
            gradioContainer.style.position = 'relative';
            gradioContainer.style.paddingTop = '60px';
            gradioContainer.insertBefore(container, gradioContainer.firstChild);

            setTimeout(function() {{
                container.style.left = '50%';
                container.style.transform = 'translateX(-50%)';
            }}, 100);

            {js_dark_mode}
            return 'Animation created';
        }}
    """
    
    # Create Gradio Blocks
    with gr.Blocks(js=js, theme=theme, title="Beatrice Voice Training") as demo:
        
        # Header
        gr.Markdown(f"""
        # 🎙️ Beatrice Voice Changer Training WebUI
        
        **Environment:** {'🚀 Kaggle' if IN_KAGGLE else '🚀 Colab' if IN_COLAB else '🖥️ Local'}
        
        Train custom voice models for real-time voice conversion.
        """)
        
        # Tab 1: Create Dataset
        with gr.Tab("📁 Create Dataset"):
            gr.Markdown("""
            ### Step 1: Prepare Training Dataset
            
            1. Upload a dataset folder containing **subfolders** (one per speaker)
            2. Each subfolder should contain audio files of that speaker
            3. Click **Begin Process** to transcribe and split audio files
            """)
            
            with gr.Row():
                with gr.Column():
                    hidden_dataset_textbox = gr.Textbox(value=str(Config.DATASETS_DIR), visible=False)
                    list_of_datasets = get_available_items(str(Config.DATASETS_DIR), directory_only=True)
                    
                    folder_to_process = gr.Dropdown(
                        choices=list_of_datasets,
                        value=None,
                        label="📂 Dataset to Process",
                        info="Select a dataset folder to process"
                    )
                    
                    with gr.Row():
                        refresh_datasets_button = gr.Button("🔄 Refresh Datasets", size="sm")
                        move_existing_folder_button = gr.Button("📦 Move Existing Folder", size="sm")
                    
                    process_button = gr.Button("▶️ Begin Process", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    #### 💡 Tips:
                    - Use high-quality audio (16kHz+ sample rate)
                    - Each speaker folder should have 10+ minutes of speech
                    - Clear speech works best (minimal background noise)
                    """)
                
                with gr.Column():
                    console_output = gr.Textbox(
                        label="📊 Progress Console",
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
            
            # Event handlers for Create Dataset
            process_button.click(
                fn=process_proxy,
                inputs=folder_to_process,
                outputs=console_output
            )
            
            folder_to_process.change(
                fn=folder_to_process_proxy,
                inputs=folder_to_process,
                outputs=folder_to_process
            )
            
            destination_root = gr.Textbox(
                value=str(Config.TRAINING_DIR / "moved_training_datasets"), 
                visible=False
            )
            source_root = gr.Textbox(value=str(Config.TRAINING_DIR), visible=False)
            
            move_existing_folder_button.click(
                fn=move_existing_folder,
                inputs=[source_root, folder_to_process, destination_root]
            )
        
        # Tab 2: Train
        with gr.Tab("🎯 Train"):
            gr.Markdown("""
            ### Step 2: Train Voice Model
            
            Configure training parameters and start training your voice model.
            """)
            
            with gr.Row():
                with gr.Column():
                    hidden_train_textbox = gr.Textbox(value=str(Config.TRAINING_DIR), visible=False)
                    
                    TRAINING_SETTINGS = {}
                    list_of_training_datasets = get_available_items(str(Config.TRAINING_DIR), directory_only=True)
                    
                    TRAINING_SETTINGS["dataset_name"] = gr.Dropdown(
                        label="📂 Dataset to Train",
                        choices=list_of_training_datasets,
                        value=list_of_training_datasets[0] if list_of_training_datasets else '',
                        info="Select a processed dataset"
                    )
                    
                    refresh_training_available_button = gr.Button("🔄 Refresh Training Datasets")
                    
                    TRAINING_SETTINGS["batch_size"] = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=64,
                        value=Config.DEFAULT_BATCH_SIZE,
                        step=1,
                        info="Higher = faster but more VRAM"
                    )
                    
                    TRAINING_SETTINGS["epochs"] = gr.Slider(
                        label="Number of Epochs",
                        minimum=1,
                        maximum=1000,
                        value=Config.DEFAULT_EPOCHS,
                        step=1,
                        info="Complete passes through dataset"
                    )
                    
                    TRAINING_SETTINGS["num_workers"] = gr.Slider(
                        label="Number of Workers",
                        minimum=1,
                        maximum=32,
                        value=Config.DEFAULT_NUM_WORKERS,
                        step=1,
                        info="Data loading workers"
                    )
                    
                    TRAINING_SETTINGS["save_interval"] = gr.Slider(
                        label="Save Interval (Epochs)",
                        minimum=1,
                        maximum=200,
                        value=Config.DEFAULT_SAVE_INTERVAL,
                        step=1,
                        info="Save model every N epochs"
                    )
                    
                    TRAINING_SETTINGS["log_interval"] = gr.Slider(
                        label="Console Log Interval",
                        minimum=10,
                        maximum=1000,
                        value=Config.DEFAULT_LOG_INTERVAL,
                        step=10,
                        info="Log metrics every N steps"
                    )
                    
                    TRAINING_SETTINGS["resume"] = gr.Checkbox(
                        label="Resume Training",
                        value=False,
                        info="Resume from last checkpoint"
                    )
                
                with gr.Column():
                    recommendation_console = gr.Textbox(
                        label="💡 Jarod's Recommendation",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                output_console = gr.Textbox(
                    label="📊 Training Console",
                    lines=10,
                    max_lines=15,
                    interactive=False
                )
            
            with gr.Row():
                with gr.Column():
                    start_train_button = gr.Button("🚀 Start Training", variant="primary", size="lg")
                with gr.Column():
                    launch_tb_button = gr.Button("📈 Launch Tensorboard", size="lg")
            
            with gr.Row():
                gr.HTML("""
                <div style="padding: 20px; background: rgba(0,0,0,0.1); border-radius: 10px;">
                    <h2>📚 Training Guide</h2>
                    
                    <h3>What are Batches?</h3>
                    <p>Groups of audio files processed together before updating the model. Larger batches = faster training but more VRAM usage.</p>
                    
                    <h3>What are Epochs?</h3>
                    <p>One complete pass through your entire dataset. More epochs = better quality but longer training time.</p>
                    
                    <h3>What are Workers?</h3>
                    <p>Parallel processes that prepare your data. More workers = faster data loading but more CPU/RAM usage.</p>
                    
                    <h3>💡 Recommendations:</h3>
                    <ul>
                        <li><b>Small datasets (< 1000 files):</b> Higher epochs (50+), lower save interval (5)</li>
                        <li><b>Large datasets (> 5000 files):</b> Lower epochs (20-30), higher save interval (10)</li>
                        <li><b>Kaggle/Colab:</b> Use batch size 8-16 for optimal GPU usage</li>
                        <li><b>Log interval:</b> 10-100 for good monitoring without spam</li>
                    </ul>
                </div>
                """)
            
            # Event handlers for recommendations
            for key, component in TRAINING_SETTINGS.items():
                if isinstance(component, gr.Dropdown):
                    component.change(
                        fn=recommendation_proxy,
                        inputs=[TRAINING_SETTINGS["dataset_name"], TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                    )
                elif isinstance(component, gr.Slider):
                    component.release(
                        fn=recommendation_proxy,
                        inputs=[TRAINING_SETTINGS["dataset_name"], TRAINING_SETTINGS["epochs"]],
                        outputs=recommendation_console
                    )
            
            # Training button
            start_train_button.click(
                fn=training_proxy,
                inputs=[
                    TRAINING_SETTINGS["dataset_name"],
                    TRAINING_SETTINGS["batch_size"],
                    TRAINING_SETTINGS["epochs"],
                    TRAINING_SETTINGS["num_workers"],
                    TRAINING_SETTINGS["resume"],
                    TRAINING_SETTINGS["save_interval"],
                    TRAINING_SETTINGS["log_interval"]
                ],
                outputs=output_console
            )
            
            # Tensorboard button
            launch_tb_button.click(fn=launch_tensorboard_proxy)
            
            # Refresh button
            hidden_option1 = gr.Textbox(value="directory", visible=False)
            hidden_extensions1 = gr.Textbox(value="[]", visible=False)
            
            refresh_training_available_button.click(
                fn=refresh_dropdown_proxy,
                inputs=[hidden_train_textbox, hidden_extensions1, hidden_option1],
                outputs=[TRAINING_SETTINGS["dataset_name"]]
            )
        
        # Tab 3: Settings
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### UI Theme Settings")
            
            with gr.Row():
                dark_mode_btn = gr.Button("🌙 Toggle Dark Mode", variant="primary")
                toggle_theme_btn = gr.Button("🎨 Toggle Custom Theme", variant="primary")
            
            gr.Markdown("""
            #### 💡 Note:
            Theme changes will take effect after restarting the WebUI.
            """)
            
            toggle_theme_btn.click(toggle_theme)
            dark_mode_btn.click(toggle_dark_mode)
            
            # Dark mode toggle JavaScript
            dark_mode_btn.click(
                None,
                None,
                None,
                js="""() => {
                    if (document.querySelectorAll('.dark').length) {
                        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                    } else {
                        document.querySelector('body').classList.add('dark');
                    }
                }""",
                show_api=False
            )
            
            # Also refresh datasets dropdown
            hidden_dataset_refresh = gr.Textbox(value=str(Config.DATASETS_DIR), visible=False)
            
            refresh_datasets_button.click(
                fn=refresh_dropdown_proxy,
                inputs=[hidden_dataset_textbox, hidden_extensions1, hidden_option1],
                outputs=[folder_to_process]
            )
    
    return demo

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    # Import heavy libraries only when running
    try:
        import whisperx
        from whisperx.utils import get_writer
        import torch
        
        print(f"\n{'='*80}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*80}\n")
        
    except ImportError as e:
        print(f"⚠️ Warning: Some dependencies are missing: {e}")
        print("Please install required packages:")
        print("pip install whisperx torch torchaudio gradio pydub pysrt")
    
    # Initialize global whisper model variable
    whisper_model = None
    
    # Create UI
    demo = create_ui()
    
    # Launch settings
    launch_kwargs = {
        'server_name': Config.GRADIO_SERVER_NAME,
        'server_port': Config.GRADIO_SERVER_PORT,
        'share': Config.GRADIO_SHARE,
        'inbrowser': Config.GRADIO_INBROWSER,
        'show_error': True,
        'debug': True
    }
    
    print(f"\n{'='*80}")
    print(f"🚀 Launching Beatrice Training WebUI")
    print(f"{'='*80}")
    print(f"Server: {launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
    print(f"Share: {launch_kwargs['share']}")
    print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'Local'}")
    print(f"{'='*80}\n")
    
    # Launch
    demo.launch(**launch_kwargs)
