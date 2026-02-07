"""
================================================================================
LIVE FACE ANALYZER WITH ADVERSARIAL ATTACKS
================================================================================
Real-time face detection, recognition, and adversarial attack demonstration.

Features:
- YuNet face detection
- SFace face recognition (identity matching)
- Live adversarial patch injection
- Live Gaussian noise injection
- Real-time visualization of attack effects

Controls:
    'a' - Toggle Adversarial Patch (ROA-style)
    'n' - Toggle Gaussian Noise
    'p' - Change patch position (center/forehead/eyes)
    '+' - Increase attack intensity
    '-' - Decrease attack intensity
    's' - Save current frame
    'q' - Quit

Author: ARAVM Framework
================================================================================
"""

import cv2
import numpy as np
import pickle
import yaml
import time
import os
from pathlib import Path
from datetime import datetime


class LiveFaceAnalyzer:
    """
    Real-time face detection, recognition, and adversarial attack demo
    """
    
    def __init__(
        self,
        detector_model: str = "facedetection/models/face_detection_yunet_2023mar.onnx",
        recognizer_model: str = "facedetection/models/face_recognition_sface_2021dec.onnx",
        embeddings_file: str = "facedetection/embeddings.pkl",
        camera_id: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        similarity_threshold: float = 0.4,
        score_threshold: float = 0.9
    ):
        """
        Initialize live face analyzer
        
        Args:
            detector_model: Path to YuNet ONNX model
            recognizer_model: Path to SFace ONNX model
            embeddings_file: Path to stored group member embeddings
            camera_id: Webcam device ID (0 = default)
            frame_width: Camera frame width
            frame_height: Camera frame height
            similarity_threshold: Face matching threshold
            score_threshold: Detection confidence threshold
        """
        self.detector_model = detector_model
        self.recognizer_model = recognizer_model
        self.embeddings_file = embeddings_file
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.similarity_threshold = similarity_threshold
        self.score_threshold = score_threshold
        
        # Attack state
        self.patch_attack_active = False
        self.noise_attack_active = False
        self.patch_position = "center"  # center, forehead, eyes
        self.attack_intensity = 0.3  # 0.1 to 1.0
        self.noise_epsilon = 0.05  # Noise intensity
        
        # Adversarial patch (generated or loaded)
        self.adversarial_patch = None
        
        # Initialize components
        self._init_models()
        self._load_embeddings()
        self._generate_adversarial_patch()
        
        print("\n" + "="*60)
        print("  LIVE FACE ANALYZER - READY")
        print("="*60)
        print("\nControls:")
        print("  [a] Toggle Adversarial Patch")
        print("  [n] Toggle Gaussian Noise")
        print("  [p] Change patch position")
        print("  [+] Increase attack intensity")
        print("  [-] Decrease attack intensity")
        print("  [s] Save screenshot")
        print("  [q] Quit")
        print("="*60 + "\n")
    
    def _init_models(self):
        """Initialize YuNet detector and SFace recognizer"""
        
        # Check if model files exist
        if not Path(self.detector_model).exists():
            print(f"⚠ Detector model not found: {self.detector_model}")
            print("  Downloading models...")
            self._download_models()
        
        if not Path(self.recognizer_model).exists():
            print(f"⚠ Recognizer model not found: {self.recognizer_model}")
            print("  Downloading models...")
            self._download_models()
        
        # Initialize YuNet face detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.detector_model,
            config="",
            input_size=(self.frame_width, self.frame_height),
            score_threshold=self.score_threshold,
            nms_threshold=0.3,
            top_k=5000
        )
        print(f"✓ YuNet Face Detector initialized")
        
        # Initialize SFace recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.recognizer_model,
            config=""
        )
        print(f"✓ SFace Face Recognizer initialized")
    
    def _download_models(self):
        """Download YuNet and SFace models if missing"""
        import urllib.request
        import ssl
        
        # SSL fix for macOS
        ssl._create_default_https_context = ssl._create_unverified_context
        
        models_dir = Path(self.detector_model).parent
        models_dir.mkdir(parents=True, exist_ok=True)
        
        models = {
            "face_detection_yunet_2023mar.onnx": 
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "face_recognition_sface_2021dec.onnx":
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        }
        
        for filename, url in models.items():
            filepath = models_dir / filename
            if not filepath.exists():
                print(f"  Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, str(filepath))
                    print(f"  ✓ Downloaded: {filename}")
                except Exception as e:
                    print(f"  ✗ Failed to download {filename}: {e}")
    
    def _load_embeddings(self):
        """Load group member embeddings from file"""
        self.embeddings_db = {
            "names": [],
            "embeddings": [],
            "metadata": {}
        }
        
        embeddings_path = Path(self.embeddings_file)
        
        if not embeddings_path.exists():
            print(f"⚠ No embeddings file found: {self.embeddings_file}")
            print("  Running in detection-only mode (no identity matching)")
            return
        
        try:
            if embeddings_path.suffix == '.pkl':
                with open(embeddings_path, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
            elif embeddings_path.suffix in ['.yml', '.yaml']:
                with open(embeddings_path, 'r') as f:
                    data = yaml.safe_load(f)
                self.embeddings_db = {
                    "names": data.get("names", []),
                    "embeddings": [np.array(emb) for emb in data.get("embeddings", [])],
                    "metadata": data.get("metadata", {})
                }
            
            print(f"✓ Loaded {len(self.embeddings_db['names'])} group member(s)")
            for name in self.embeddings_db['names']:
                print(f"    - {name}")
        except Exception as e:
            print(f"⚠ Could not load embeddings: {e}")
    
    def _generate_adversarial_patch(self):
        """Generate a colorful adversarial-looking patch"""
        # Create a visually striking patch (can be replaced with trained patch)
        patch_size = 100
        
        # Generate noise-like colorful pattern
        np.random.seed(42)  # Reproducible
        patch = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
        
        # Add some structure for visual effect
        cv2.circle(patch, (50, 50), 30, (0, 255, 0), 3)
        cv2.circle(patch, (50, 50), 20, (255, 0, 0), 3)
        cv2.circle(patch, (50, 50), 10, (0, 0, 255), -1)
        
        # Add some noise bands
        for i in range(0, patch_size, 10):
            color = (np.random.randint(0, 255), 
                    np.random.randint(0, 255), 
                    np.random.randint(0, 255))
            cv2.line(patch, (0, i), (patch_size, i), color, 2)
        
        self.adversarial_patch = patch
        print("✓ Adversarial patch generated")
        
        # Try to load a real patch from phattacks if available
        patch_paths = [
            "phattacks/adversarial_patch.png",
            "phattacks/patch.png",
            "face/patches/attack_patch.png"
        ]
        
        for patch_path in patch_paths:
            if Path(patch_path).exists():
                loaded_patch = cv2.imread(patch_path)
                if loaded_patch is not None:
                    self.adversarial_patch = loaded_patch
                    print(f"✓ Loaded adversarial patch from: {patch_path}")
                    break
    
    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between embeddings"""
        return np.dot(emb1.flatten(), emb2.flatten()) / \
               (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    
    def recognize_face(self, embedding):
        """Match embedding against database"""
        if len(self.embeddings_db["names"]) == 0:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_similarity = 0.0
        
        for name, stored_emb in zip(self.embeddings_db["names"], 
                                    self.embeddings_db["embeddings"]):
            similarity = self.cosine_similarity(embedding, stored_emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity < self.similarity_threshold:
            return "Unknown", best_similarity
        
        return best_match, best_similarity
    
    def apply_adversarial_patch(self, frame, face_bbox):
        """Apply adversarial patch to face region"""
        x, y, w, h = face_bbox[:4].astype(int)
        
        # Calculate patch size based on face size and intensity
        patch_size = int(min(w, h) * self.attack_intensity)
        patch_resized = cv2.resize(self.adversarial_patch, (patch_size, patch_size))
        
        # Calculate patch position
        if self.patch_position == "center":
            px = x + w // 2 - patch_size // 2
            py = y + h // 2 - patch_size // 2
        elif self.patch_position == "forehead":
            px = x + w // 2 - patch_size // 2
            py = y + int(h * 0.1)
        elif self.patch_position == "eyes":
            px = x + w // 2 - patch_size // 2
            py = y + int(h * 0.35)
        else:
            px = x + w // 2 - patch_size // 2
            py = y + h // 2 - patch_size // 2
        
        # Ensure within bounds
        px = max(0, min(px, frame.shape[1] - patch_size))
        py = max(0, min(py, frame.shape[0] - patch_size))
        
        # Apply patch with some blending
        alpha = 0.85
        try:
            roi = frame[py:py+patch_size, px:px+patch_size]
            if roi.shape[:2] == patch_resized.shape[:2]:
                frame[py:py+patch_size, px:px+patch_size] = \
                    cv2.addWeighted(patch_resized, alpha, roi, 1-alpha, 0)
        except:
            pass
        
        return frame
    
    def apply_gaussian_noise(self, frame, face_bbox):
        """Apply Gaussian noise to face region (FGSM-style)"""
        x, y, w, h = face_bbox[:4].astype(int)
        
        # Ensure bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        # Extract face region
        face_roi = frame[y:y2, x:x2].copy()
        
        # Generate noise
        noise = np.random.normal(0, 255 * self.noise_epsilon * self.attack_intensity, 
                                face_roi.shape).astype(np.float32)
        
        # Add noise to face
        noisy_face = face_roi.astype(np.float32) + noise
        noisy_face = np.clip(noisy_face, 0, 255).astype(np.uint8)
        
        # Put back
        frame[y:y2, x:x2] = noisy_face
        
        return frame
    
    def draw_status_panel(self, frame):
        """Draw attack status panel on frame"""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Status text
        y_offset = 30
        
        # Title
        cv2.putText(frame, "ARAVM Live Analyzer", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        # Attack status
        if self.patch_attack_active:
            status = f"PATCH ATTACK: ON ({self.patch_position})"
            color = (0, 0, 255)
        elif self.noise_attack_active:
            status = f"NOISE ATTACK: ON (eps={self.noise_epsilon:.2f})"
            color = (0, 165, 255)
        else:
            status = "NO ATTACK"
            color = (0, 255, 0)
        
        cv2.putText(frame, status, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += 20
        
        # Intensity
        cv2.putText(frame, f"Intensity: {self.attack_intensity:.1%}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # FPS placeholder
        cv2.putText(frame, "Press [q] to quit", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def draw_face_info(self, frame, face, name, similarity, attacked=False):
        """Draw bounding box and identity info on face"""
        x, y, w, h = face[:4].astype(int)
        
        # Color based on recognition and attack state
        if attacked:
            if name == "Unknown":
                color = (0, 0, 255)  # Red - attack successful
            else:
                color = (0, 165, 255)  # Orange - recognized under attack
        else:
            if name == "Unknown":
                color = (128, 128, 128)  # Gray
            else:
                color = (0, 255, 0)  # Green - recognized
        
        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Label background
        label = f"{name} ({similarity:.2f})"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y-25), (x + label_w + 10, y), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x+5, y-7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Attack indicator
        if attacked:
            cv2.putText(frame, "⚠ ATTACKED", (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop - run live face analyzer"""
        
        # Open webcam
        print(f"\n📷 Opening webcam (ID: {self.camera_id})...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("✗ ERROR: Could not open webcam!")
            print("  Possible solutions:")
            print("  1. Check if camera is connected")
            print("  2. Check if another app is using the camera")
            print("  3. Try a different camera ID (e.g., 1)")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera opened: {actual_width}x{actual_height}")
        
        # Update detector input size if resolution changed
        if actual_width != self.frame_width or actual_height != self.frame_height:
            self.detector.setInputSize((actual_width, actual_height))
        
        print("\n🎬 Starting live analysis... Press 'q' to quit.\n")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Mirror for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Create display frame (may be modified by attacks)
            display_frame = frame.copy()
            recognition_frame = frame.copy()  # Clean frame for recognition
            
            # Detect faces
            _, faces = self.detector.detect(frame)
            
            attacked = self.patch_attack_active or self.noise_attack_active
            
            if faces is not None and len(faces) > 0:
                for face in faces:
                    # Apply attacks to display frame
                    if self.patch_attack_active:
                        display_frame = self.apply_adversarial_patch(display_frame, face)
                        # Also attack the recognition frame to show real effect
                        recognition_frame = self.apply_adversarial_patch(recognition_frame, face)
                    
                    if self.noise_attack_active:
                        display_frame = self.apply_gaussian_noise(display_frame, face)
                        recognition_frame = self.apply_gaussian_noise(recognition_frame, face)
                    
                    # Extract features and recognize from attacked frame
                    try:
                        aligned_face = self.recognizer.alignCrop(recognition_frame, face)
                        embedding = self.recognizer.feature(aligned_face)
                        name, similarity = self.recognize_face(embedding)
                    except Exception as e:
                        name = "Error"
                        similarity = 0.0
                    
                    # Draw face info
                    display_frame = self.draw_face_info(
                        display_frame, face, name, similarity, attacked
                    )
            
            # Draw status panel
            display_frame = self.draw_status_panel(display_frame)
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (self.frame_width - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show frame
            cv2.imshow("ARAVM Live Face Analyzer", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Exiting...")
                break
            
            elif key == ord('a'):
                self.patch_attack_active = not self.patch_attack_active
                self.noise_attack_active = False  # Only one attack at a time
                status = "ON" if self.patch_attack_active else "OFF"
                print(f"🎯 Adversarial Patch: {status}")
            
            elif key == ord('n'):
                self.noise_attack_active = not self.noise_attack_active
                self.patch_attack_active = False  # Only one attack at a time
                status = "ON" if self.noise_attack_active else "OFF"
                print(f"📊 Gaussian Noise Attack: {status}")
            
            elif key == ord('p'):
                positions = ["center", "forehead", "eyes"]
                current_idx = positions.index(self.patch_position)
                self.patch_position = positions[(current_idx + 1) % len(positions)]
                print(f"📍 Patch position: {self.patch_position}")
            
            elif key == ord('+') or key == ord('='):
                self.attack_intensity = min(1.0, self.attack_intensity + 0.1)
                print(f"⬆️ Attack intensity: {self.attack_intensity:.1%}")
            
            elif key == ord('-'):
                self.attack_intensity = max(0.1, self.attack_intensity - 0.1)
                print(f"⬇️ Attack intensity: {self.attack_intensity:.1%}")
            
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Live analyzer closed.")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  LIVE FACE ANALYZER - Adversarial Robustness Demo")
    print("="*60 + "\n")
    
    # Initialize and run
    try:
        analyzer = LiveFaceAnalyzer(
            detector_model="facedetection/models/face_detection_yunet_2023mar.onnx",
            recognizer_model="facedetection/models/face_recognition_sface_2021dec.onnx",
            embeddings_file="facedetection/embeddings.pkl",
            camera_id=0,
            frame_width=640,
            frame_height=480,
            similarity_threshold=0.4
        )
        
        analyzer.run()
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure the model files exist:")
        print("  1. facedetection/models/face_detection_yunet_2023mar.onnx")
        print("  2. facedetection/models/face_recognition_sface_2021dec.onnx")
        print("\nRun this to download: python facedetection/download_models.py")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
