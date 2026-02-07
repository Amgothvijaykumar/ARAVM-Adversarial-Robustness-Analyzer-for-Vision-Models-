"""
Download YuNet and SFace model weights automatically
"""
import os
import urllib.request
from pathlib import Path

def download_file(url, destination):
    """Download file with progress indication"""
    print(f"Downloading {destination}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded: {destination}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {destination}: {e}")
        return False

def download_models():
    """Download YuNet and SFace model weights"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    models = {
        "face_detection_yunet_2023mar.onnx": 
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx":
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    }
    
    for filename, url in models.items():
        filepath = models_dir / filename
        if filepath.exists():
            print(f"Model already exists: {filepath}")
        else:
            download_file(url, str(filepath))
    
    print("\n✓ All models ready!")
    return models_dir

if __name__ == "__main__":
    download_models()