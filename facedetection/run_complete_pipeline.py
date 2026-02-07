"""
Complete pipeline: Download models, train, and test
"""
import sys
from pathlib import Path

def main():
    print("="*70)
    print(" Face Recognition System - Complete Pipeline")
    print(" YuNet (Detection) + SFace (Recognition) + Adversarial Testing")
    print("="*70 + "\n")
    
    # Step 1: Download models
    print("STEP 1: Downloading models...")
    print("-"*70)
    from download_models import download_models
    models_dir = download_models()
    print()
    
    # Step 2: Train (extract embeddings)
    print("\nSTEP 2: Training (Extracting embeddings)...")
    print("-"*70)
    
    if not Path("group_members").exists():
        print("⚠ Please create 'group_members' directory with structure:")
        print("  group_members/")
        print("    ├── alice/")
        print("    │   ├── alice1.jpg")
        print("    │   └── alice2.jpg")
        print("    ├── bob/")
        print("    │   └── bob1.jpg")
        print("\nCreate this directory and run again.")
        return
    
    from train_face_recognition import FaceRecognitionTrainer
    trainer = FaceRecognitionTrainer()
    trainer.train_from_directory(
        group_members_dir="group_members",
        output_file="embeddings.pkl"
    )
    print()
    
    # Step 3: Test inference
    print("\nSTEP 3: Testing inference...")
    print("-"*70)
    
    test_image = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if test_image and Path(test_image).exists():
        from inference_face_recognition import FaceRecognizer
        recognizer = FaceRecognizer(embeddings_file="embeddings.pkl")
        recognizer.process_image(test_image, output_path="result_labeled.png")
    else:
        print("Skipping inference test")
    print()
    
    # Step 4: Adversarial testing (optional)
    print("\nSTEP 4: Adversarial testing (optional)...")
    print("-"*70)
    
    do_adversarial = input("Test adversarial patch attack? (y/n): ").strip().lower()
    
    if do_adversarial == 'y':
        if not Path("phattacks").exists():
            print("Creating 'phattacks' directory...")
            Path("phattacks").mkdir(exist_ok=True)
            print("⚠ Please add adversarial patch images to 'phattacks/' directory")
            return
        
        victim_img = input("Enter path to victim image: ").strip()
        patch_name = input("Enter patch filename (in phattacks/): ").strip()
        
        if Path(victim_img).exists():
            from adversarial_patch_attack import AdversarialPatchTester
            from inference_face_recognition import FaceRecognizer
            
            recognizer = FaceRecognizer(embeddings_file="embeddings.pkl")
            tester = AdversarialPatchTester(recognizer, patch_attacks_dir="phattacks")
            
            tester.test_patch_attack(
                original_image_path=victim_img,
                patch_name=patch_name,
                position="forehead",
                scale=0.3
            )
    
    print("\n" + "="*70)
    print(" Pipeline Complete!")
    print("="*70)

if __name__ == "__main__":
    main()