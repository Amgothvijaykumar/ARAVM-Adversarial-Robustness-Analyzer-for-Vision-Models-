"""
Training Phase: Extract face embeddings from group members
"""
import cv2
import numpy as np
import pickle
from pathlib import Path
import yaml
import os

# Get the directory where this script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

class FaceRecognitionTrainer:
    def __init__(self, 
                 detector_model=None,
                 recognizer_model=None,
                 score_threshold=0.9,
                 nms_threshold=0.3,
                 top_k=5000):
        """
        Initialize face detection and recognition models
        
        Args:
            detector_model: Path to YuNet ONNX model
            recognizer_model: Path to SFace ONNX model
            score_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            top_k: Maximum number of detections
        """
        # Use default paths relative to script directory if not specified
        if detector_model is None:
            detector_model = str(SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx")
        if recognizer_model is None:
            recognizer_model = str(SCRIPT_DIR / "models" / "face_recognition_sface_2021dec.onnx")
        
        self.detector_model = detector_model
        self.recognizer_model = recognizer_model
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        
        # Initialize detector (will be set per image due to size requirements)
        self.detector = None
        
        # Initialize recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=recognizer_model,
            config=""
        )
        
        print("✓ Models initialized successfully")
    
    def _init_detector(self, img_width, img_height):
        """Initialize detector with specific image dimensions"""
        self.detector = cv2.FaceDetectorYN.create(
            model=self.detector_model,
            config="",
            input_size=(img_width, img_height),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )
    
    def detect_face(self, image):
        """
        Detect face in image using YuNet
        
        Returns:
            face: Detected face region or None
            bbox: Bounding box [x, y, w, h] or None
        """
        h, w = image.shape[:2]
        self._init_detector(w, h)
        
        # Detect faces
        _, faces = self.detector.detect(image)
        
        if faces is None or len(faces) == 0:
            return None, None
        
        # Take the first (most confident) face
        face = faces[0]
        bbox = face[:4].astype(int)  # x, y, w, h
        
        return face, bbox
    
    def extract_features(self, image, face):
        """
        Extract face embedding using SFace
        
        Args:
            image: Original image
            face: Face detection result from YuNet
            
        Returns:
            feature: 128-d embedding vector
        """
        aligned_face = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned_face)
        return feature
    
    def train_from_directory(self, group_members_dir="group_members", output_file="embeddings.pkl"):
        """
        Train by extracting embeddings from all images in directory
        
        Directory structure expected:
        group_members/
            ├── person1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            ├── person2/
            │   ├── img1.jpg
            
        Args:
            group_members_dir: Root directory containing person subdirectories
            output_file: Output file for storing embeddings (.pkl or .yml)
        """
        members_path = Path(group_members_dir)
        
        if not members_path.exists():
            print(f"✗ Directory not found: {group_members_dir}")
            print("Please create the directory structure:")
            print("  group_members/")
            print("    ├── person1/")
            print("    │   ├── image1.jpg")
            print("    ├── person2/")
            print("    │   ├── image1.jpg")
            return
        
        embeddings_db = {
            "names": [],
            "embeddings": [],
            "metadata": []
        }
        
        # Iterate through person directories
        person_dirs = [d for d in members_path.iterdir() if d.is_dir()]
        
        if len(person_dirs) == 0:
            print(f"✗ No person directories found in {group_members_dir}")
            return
        
        print(f"\n🔍 Processing {len(person_dirs)} person(s)...\n")
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            print(f"Processing: {person_name}")
            
            # Get all image files
            image_files = list(person_dir.glob("*.jpg")) + \
                         list(person_dir.glob("*.jpeg")) + \
                         list(person_dir.glob("*.png"))
            
            if len(image_files) == 0:
                print(f"  ⚠ No images found for {person_name}")
                continue
            
            person_embeddings = []
            
            for img_path in image_files:
                try:
                    # Read image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"  ✗ Could not read: {img_path.name}")
                        continue
                    
                    # Detect face
                    face, bbox = self.detect_face(image)
                    
                    if face is None:
                        print(f"  ✗ No face detected: {img_path.name}")
                        continue
                    
                    # Extract features
                    embedding = self.extract_features(image, face)
                    person_embeddings.append(embedding)
                    
                    print(f"  ✓ Processed: {img_path.name} - Face at {bbox}")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {img_path.name}: {e}")
                    continue
            
            if len(person_embeddings) > 0:
                # Average embeddings if multiple images per person
                avg_embedding = np.mean(person_embeddings, axis=0)
                
                embeddings_db["names"].append(person_name)
                embeddings_db["embeddings"].append(avg_embedding)
                embeddings_db["metadata"].append({
                    "num_images": len(person_embeddings),
                    "source_dir": str(person_dir)
                })
                
                print(f"  ✓ Stored embedding for {person_name} (from {len(person_embeddings)} image(s))\n")
            else:
                print(f"  ⚠ No valid embeddings for {person_name}\n")
        
        # Save embeddings
        if len(embeddings_db["names"]) > 0:
            self.save_embeddings(embeddings_db, output_file)
            print(f"\n✓ Training complete! Stored {len(embeddings_db['names'])} person(s)")
        else:
            print("\n✗ No embeddings extracted. Please check your images.")
    
    def save_embeddings(self, embeddings_db, output_file):
        """Save embeddings to file"""
        output_path = Path(output_file)
        
        if output_path.suffix == '.pkl':
            with open(output_file, 'wb') as f:
                pickle.dump(embeddings_db, f)
            print(f"✓ Saved embeddings to {output_file} (pickle format)")
            
        elif output_path.suffix in ['.yml', '.yaml']:
            # Convert numpy arrays to lists for YAML
            yaml_data = {
                "names": embeddings_db["names"],
                "embeddings": [emb.tolist() for emb in embeddings_db["embeddings"]],
                "metadata": embeddings_db["metadata"]
            }
            with open(output_file, 'w') as f:
                yaml.dump(yaml_data, f)
            print(f"✓ Saved embeddings to {output_file} (YAML format)")
        else:
            print(f"✗ Unsupported format: {output_path.suffix}. Use .pkl or .yml")

def main():
    """Main training function"""
    print("="*60)
    print("Face Recognition Training - YuNet + SFace")
    print("="*60)
    
    trainer = FaceRecognitionTrainer()
    
    # Use paths relative to script directory
    group_members_dir = str(SCRIPT_DIR / "group_members")
    output_file = str(SCRIPT_DIR / "embeddings.pkl")
    
    trainer.train_from_directory(
        group_members_dir=group_members_dir,
        output_file=output_file
    )

if __name__ == "__main__":
    main()