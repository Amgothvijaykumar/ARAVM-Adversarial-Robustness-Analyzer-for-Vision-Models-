"""
Inference Phase: Recognize faces in new images
"""
import cv2
import numpy as np
import pickle
import yaml
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

class FaceRecognizer:
    def __init__(self,
                 embeddings_file=None,
                 detector_model=None,
                 recognizer_model=None,
                 similarity_threshold=0.4,
                 score_threshold=0.9):
        """
        Initialize face recognizer
        
        Args:
            embeddings_file: Path to stored embeddings
            detector_model: Path to YuNet model
            recognizer_model: Path to SFace model
            similarity_threshold: Cosine similarity threshold for recognition
            score_threshold: Detection confidence threshold
        """
        # Use default paths relative to script directory if not specified
        if embeddings_file is None:
            embeddings_file = str(SCRIPT_DIR / "embeddings.pkl")
        if detector_model is None:
            detector_model = str(SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx")
        if recognizer_model is None:
            recognizer_model = str(SCRIPT_DIR / "models" / "face_recognition_sface_2021dec.onnx")
        
        self.detector_model = detector_model
        self.recognizer_model = recognizer_model
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        
        # Load embeddings database
        self.embeddings_db = self.load_embeddings(embeddings_file)
        
        # Initialize detector
        self.detector = None
        
        # Initialize recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=recognizer_model,
            config=""
        )
        
        print(f"✓ Loaded {len(self.embeddings_db['names'])} person(s) from database")
    
    def load_embeddings(self, embeddings_file):
        """Load embeddings from file"""
        file_path = Path(embeddings_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        if file_path.suffix == '.pkl':
            with open(embeddings_file, 'rb') as f:
                embeddings_db = pickle.load(f)
        elif file_path.suffix in ['.yml', '.yaml']:
            with open(embeddings_file, 'r') as f:
                data = yaml.safe_load(f)
            # Convert lists back to numpy arrays
            embeddings_db = {
                "names": data["names"],
                "embeddings": [np.array(emb) for emb in data["embeddings"]],
                "metadata": data["metadata"]
            }
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return embeddings_db
    
    def _init_detector(self, img_width, img_height):
        """Initialize detector with specific image dimensions"""
        self.detector = cv2.FaceDetectorYN.create(
            model=self.detector_model,
            config="",
            input_size=(img_width, img_height),
            score_threshold=self.score_threshold,
            nms_threshold=0.3,
            top_k=5000
        )
    
    def detect_faces(self, image):
        """Detect all faces in image"""
        h, w = image.shape[:2]
        self._init_detector(w, h)
        
        _, faces = self.detector.detect(image)
        return faces
    
    def extract_features(self, image, face):
        """Extract embedding from detected face"""
        aligned_face = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned_face)
        return feature
    
    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1.flatten(), embedding2.flatten()) / \
               (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def recognize_face(self, embedding):
        """
        Recognize face by comparing embedding with database
        
        Returns:
            name: Recognized person's name or "Unknown"
            similarity: Best similarity score
        """
        if len(self.embeddings_db["names"]) == 0:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for name, stored_embedding in zip(self.embeddings_db["names"], 
                                          self.embeddings_db["embeddings"]):
            similarity = self.cosine_similarity(embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Apply threshold
        if best_similarity < self.similarity_threshold:
            return "Unknown", best_similarity
        
        return best_match, best_similarity
    
    def process_image(self, image_path, output_path=None, visualize=True):
        """
        Process a single image for face recognition
        
        Args:
            image_path: Path to input image
            output_path: Path to save labeled image (optional)
            visualize: Whether to display the result
            
        Returns:
            results: List of recognition results
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"✗ Could not read image: {image_path}")
            return []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if faces is None or len(faces) == 0:
            print("✗ No faces detected")
            return []
        
        results = []
        
        # Process each face
        for i, face in enumerate(faces):
            bbox = face[:4].astype(int)
            x, y, w, h = bbox
            
            # Extract features
            embedding = self.extract_features(image, face)
            
            # Recognize
            name, similarity = self.recognize_face(embedding)
            
            results.append({
                "bbox": bbox,
                "name": name,
                "similarity": similarity
            })
            
            # Draw on image
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            label = f"{name} ({similarity:.2f})"
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            print(f"Face {i+1}: {name} (similarity: {similarity:.3f})")
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"✓ Saved labeled image to: {output_path}")
        
        # Visualize
        if visualize:
            cv2.imshow("Face Recognition", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results

def main():
    """Main inference function"""
    print("="*60)
    print("Face Recognition Inference - YuNet + SFace")
    print("="*60 + "\n")
    
    # Initialize recognizer
    recognizer = FaceRecognizer(
        embeddings_file="embeddings.pkl",
        similarity_threshold=0.4  # Adjust as needed (0.3-0.5 typical range)
    )
    
    # Process test image
    test_image = "test_image.png"  # Change to your test image path
    
    if Path(test_image).exists():
        recognizer.process_image(
            image_path=test_image,
            output_path="result_labeled.png",
            visualize=True
        )
    else:
        print(f"✗ Test image not found: {test_image}")
        print("Please provide a test image path")

if __name__ == "__main__":
    main()