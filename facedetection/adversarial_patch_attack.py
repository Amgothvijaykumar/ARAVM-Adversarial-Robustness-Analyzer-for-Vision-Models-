"""
Adversarial Attack Testing: Apply patch attacks to test robustness
"""
import cv2
import numpy as np
from pathlib import Path
from inference_face_recognition import FaceRecognizer

class AdversarialPatchTester:
    def __init__(self, recognizer, patch_attacks_dir="phattacks"):
        """
        Initialize adversarial patch tester
        
        Args:
            recognizer: FaceRecognizer instance
            patch_attacks_dir: Directory containing adversarial patches
        """
        self.recognizer = recognizer
        self.patch_attacks_dir = Path(patch_attacks_dir)
        
        if not self.patch_attacks_dir.exists():
            print(f"⚠ Warning: Patch directory not found: {patch_attacks_dir}")
            print("Creating directory...")
            self.patch_attacks_dir.mkdir(exist_ok=True)
    
    def load_patch(self, patch_name):
        """Load adversarial patch from file"""
        patch_path = self.patch_attacks_dir / patch_name
        
        if not patch_path.exists():
            print(f"✗ Patch not found: {patch_path}")
            return None
        
        patch = cv2.imread(str(patch_path), cv2.IMREAD_UNCHANGED)
        return patch
    
    def apply_patch_to_face(self, image, face_bbox, patch, position="center", scale=0.3):
        """
        Apply adversarial patch to detected face region
        
        Args:
            image: Original image
            face_bbox: Face bounding box [x, y, w, h]
            patch: Adversarial patch image
            position: Where to place patch ("center", "forehead", "eyes", "mouth")
            scale: Scale of patch relative to face size
            
        Returns:
            patched_image: Image with applied patch
        """
        x, y, w, h = face_bbox
        patched_image = image.copy()
        
        # Resize patch
        patch_w = int(w * scale)
        patch_h = int(h * scale)
        patch_resized = cv2.resize(patch, (patch_w, patch_h))
        
        # Determine patch position
        if position == "center":
            px = x + w // 2 - patch_w // 2
            py = y + h // 2 - patch_h // 2
        elif position == "forehead":
            px = x + w // 2 - patch_w // 2
            py = y + int(h * 0.1)
        elif position == "eyes":
            px = x + w // 2 - patch_w // 2
            py = y + int(h * 0.35)
        elif position == "mouth":
            px = x + w // 2 - patch_w // 2
            py = y + int(h * 0.7)
        else:
            px = x + w // 2 - patch_w // 2
            py = y + h // 2 - patch_h // 2
        
        # Ensure patch is within image bounds
        px = max(0, min(px, image.shape[1] - patch_w))
        py = max(0, min(py, image.shape[0] - patch_h))
        
        # Apply patch (handle transparency if present)
        if patch_resized.shape[2] == 4:  # RGBA
            alpha = patch_resized[:, :, 3] / 255.0
            for c in range(3):
                patched_image[py:py+patch_h, px:px+patch_w, c] = \
                    alpha * patch_resized[:, :, c] + \
                    (1 - alpha) * patched_image[py:py+patch_h, px:px+patch_w, c]
        else:  # RGB
            patched_image[py:py+patch_h, px:px+patch_w] = patch_resized
        
        return patched_image
    
    def test_patch_attack(self, 
                         original_image_path, 
                         patch_name,
                         position="center",
                         scale=0.3,
                         output_dir="attack_results"):
        """
        Test adversarial patch attack on an image
        
        Args:
            original_image_path: Path to original member image
            patch_name: Name of patch file in phattacks directory
            position: Where to place the patch
            scale: Size of patch relative to face
            output_dir: Directory to save results
        """
        print("="*60)
        print(f"Testing Adversarial Patch Attack")
        print("="*60)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Read original image
        original_image = cv2.imread(str(original_image_path))
        if original_image is None:
            print(f"✗ Could not read image: {original_image_path}")
            return
        
        # Detect face in original
        faces = self.recognizer.detect_faces(original_image)
        if faces is None or len(faces) == 0:
            print("✗ No face detected in original image")
            return
        
        face = faces[0]
        bbox = face[:4].astype(int)
        
        # Recognize original (baseline)
        print("\n1️⃣  BASELINE (No Attack):")
        embedding_original = self.recognizer.extract_features(original_image, face)
        name_original, similarity_original = self.recognizer.recognize_face(embedding_original)
        print(f"   Identity: {name_original} (similarity: {similarity_original:.3f})")
        
        # Load adversarial patch
        patch = self.load_patch(patch_name)
        if patch is None:
            return
        
        # Apply patch
        print(f"\n2️⃣  APPLYING PATCH: {patch_name}")
        print(f"   Position: {position}, Scale: {scale}")
        patched_image = self.apply_patch_to_face(original_image, bbox, patch, position, scale)
        
        # Save patched image
        patched_output = output_path / f"patched_{Path(original_image_path).name}"
        cv2.imwrite(str(patched_output), patched_image)
        print(f"   ✓ Saved patched image: {patched_output}")
        
        # Recognize patched image
        print("\n3️⃣  ATTACKING (With Patch):")
        faces_patched = self.recognizer.detect_faces(patched_image)
        
        if faces_patched is None or len(faces_patched) == 0:
            print("   ✗ No face detected after patch (detection evasion)")
            attack_success = True
            name_patched = "Not Detected"
            similarity_patched = 0.0
        else:
            face_patched = faces_patched[0]
            embedding_patched = self.recognizer.extract_features(patched_image, face_patched)
            name_patched, similarity_patched = self.recognizer.recognize_face(embedding_patched)
            print(f"   Identity: {name_patched} (similarity: {similarity_patched:.3f})")
            
            # Check if attack succeeded
            attack_success = (name_patched != name_original) or (name_patched == "Unknown")
        
        # Results summary
        print("\n" + "="*60)
        print("ATTACK RESULTS:")
        print("="*60)
        print(f"Original Identity:  {name_original} ({similarity_original:.3f})")
        print(f"Patched Identity:   {name_patched} ({similarity_patched:.3f})")
        print(f"Attack Success:     {'✓ YES' if attack_success else '✗ NO'}")
        
        if attack_success:
            if name_patched == "Unknown":
                print(f"Effect: Identity evaded to 'Unknown'")
            elif name_patched == "Not Detected":
                print(f"Effect: Face detection completely evaded")
            else:
                print(f"Effect: Identity changed from '{name_original}' to '{name_patched}'")
        
        print("="*60)
        
        # Visualize comparison
        self._visualize_comparison(original_image, patched_image, 
                                   name_original, name_patched,
                                   similarity_original, similarity_patched,
                                   output_path)
    
    def _visualize_comparison(self, original, patched, 
                             name_orig, name_patch,
                             sim_orig, sim_patch,
                             output_path):
        """Create side-by-side comparison visualization"""
        # Create comparison image
        h, w = original.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w:] = patched
        
        # Add labels
        cv2.putText(comparison, f"Original: {name_orig} ({sim_orig:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, f"Patched: {name_patch} ({sim_patch:.2f})", 
                   (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save comparison
        comparison_path = output_path / "comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"\n✓ Saved comparison image: {comparison_path}")
        
        # Display
        cv2.imshow("Adversarial Attack Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def batch_test_patches(self, image_path, patch_positions=None, scales=None):
        """
        Test multiple patch configurations
        
        Args:
            image_path: Path to victim image
            patch_positions: List of positions to test
            scales: List of scales to test
        """
        if patch_positions is None:
            patch_positions = ["center", "forehead", "eyes"]
        
        if scales is None:
            scales = [0.2, 0.3, 0.4]
        
        # Get all patches
        patches = list(self.patch_attacks_dir.glob("*.png")) + \
                 list(self.patch_attacks_dir.glob("*.jpg"))
        
        if len(patches) == 0:
            print(f"✗ No patches found in {self.patch_attacks_dir}")
            return
        
        print(f"\n🔍 Testing {len(patches)} patch(es) with {len(patch_positions)} position(s) " 
              f"and {len(scales)} scale(s)...")
        
        results = []
        
        for patch_file in patches:
            for position in patch_positions:
                for scale in scales:
                    print(f"\n--- Testing: {patch_file.name} | {position} | scale={scale} ---")
                    
                    # Run attack
                    self.test_patch_attack(
                        original_image_path=image_path,
                        patch_name=patch_file.name,
                        position=position,
                        scale=scale,
                        output_dir=f"attack_results/{patch_file.stem}_{position}_{scale}"
                    )

def main():
    """Main adversarial testing function"""
    print("="*60)
    print("Adversarial Patch Attack Testing")
    print("="*60 + "\n")
    
    # Initialize recognizer
    recognizer = FaceRecognizer(
        embeddings_file="embeddings.pkl",
        similarity_threshold=0.4
    )
    
    # Initialize adversarial tester
    tester = AdversarialPatchTester(
        recognizer=recognizer,
        patch_attacks_dir="phattacks"
    )
    
    # Test patch attack
    # Replace with actual paths
    victim_image = "group_members/person1/image1.jpg"  # Change this
    adversarial_patch = "patch1.png"  # Change this - should be in phattacks/
    
    if Path(victim_image).exists():
        tester.test_patch_attack(
            original_image_path=victim_image,
            patch_name=adversarial_patch,
            position="forehead",  # Options: center, forehead, eyes, mouth
            scale=0.3,  # Size relative to face (0.1-0.5 typical)
            output_dir="attack_results"
        )
    else:
        print(f"✗ Image not found: {victim_image}")
        print("\nPlease provide:")
        print("1. A victim image path (from group_members)")
        print("2. An adversarial patch in the 'phattacks' directory")

if __name__ == "__main__":
    main()