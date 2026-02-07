"""
================================================================================
FACE CAPTURE UTILITY
================================================================================
Captures photos of team members for face recognition training.

Usage: python capture_faces.py

Controls:
    SPACE - Take a photo
    N     - Next person
    Q     - Quit and save

================================================================================
"""

import cv2
import os
from pathlib import Path
from datetime import datetime


def capture_faces():
    """Capture faces for each team member"""
    
    base_dir = Path("facedetection/group_members")
    
    # Auto-detect team members from existing folders
    if base_dir.exists():
        team_members = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    else:
        team_members = []
    
    if len(team_members) == 0:
        print("\n❌ No team member folders found!")
        print("Create folders first:")
        print("  mkdir -p facedetection/group_members/YourName")
        return
    
    print(f"\n✅ Found {len(team_members)} team member(s): {', '.join(team_members)}")
    
    print("\n" + "="*60)
    print("  FACE CAPTURE UTILITY")
    print("="*60)
    print("\nControls:")
    print("  [SPACE] - Take a photo")
    print("  [N]     - Next person")
    print("  [Q]     - Quit")
    print("="*60)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n❌ ERROR: Could not open webcam!")
        print("Make sure no other app is using the camera.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n✓ Camera opened successfully!")
    
    current_member_idx = 0
    photos_taken = {member: 0 for member in team_members}
    
    print(f"\n📸 Now capturing: {team_members[current_member_idx]}")
    print("   Press SPACE to take photo, N for next person, Q to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Get current member
        current_member = team_members[current_member_idx]
        
        # Draw overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Status text
        cv2.putText(frame, f"Capturing: {current_member}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Photos taken: {photos_taken[current_member]}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "[SPACE]=Photo [N]=Next [Q]=Quit", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Draw face guide (center rectangle)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_size = 200
        cv2.rectangle(frame, 
                     (cx - box_size//2, cy - box_size//2 - 30),
                     (cx + box_size//2, cy + box_size//2 + 30),
                     (0, 255, 0), 2)
        cv2.putText(frame, "Position face here", (cx - 80, cy + box_size//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show frame
        cv2.imshow("Face Capture - Press SPACE to take photo", frame)
        
        # Handle key input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        
        elif key == ord(' '):  # SPACE - take photo
            # Save the original frame (without overlay)
            ret, clean_frame = cap.read()
            if ret:
                clean_frame = cv2.flip(clean_frame, 1)
                
                # Generate filename
                timestamp = datetime.now().strftime("%H%M%S")
                photo_num = photos_taken[current_member] + 1
                filename = f"{current_member}_{photo_num:02d}_{timestamp}.jpg"
                filepath = base_dir / current_member / filename
                
                # Save photo
                cv2.imwrite(str(filepath), clean_frame)
                photos_taken[current_member] += 1
                
                print(f"✅ Saved: {filepath}")
                
                # Flash effect
                flash = frame.copy()
                flash[:] = (255, 255, 255)
                cv2.imshow("Face Capture - Press SPACE to take photo", flash)
                cv2.waitKey(100)
        
        elif key == ord('n'):  # N - next person
            current_member_idx = (current_member_idx + 1) % len(team_members)
            print(f"\n📸 Now capturing: {team_members[current_member_idx]}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("  CAPTURE SUMMARY")
    print("="*60)
    total = 0
    for member, count in photos_taken.items():
        status = "✓" if count > 0 else "⚠"
        print(f"  {status} {member}: {count} photo(s)")
        total += count
    print(f"\n  Total: {total} photos captured")
    print("="*60)
    
    if total > 0:
        print("\n✅ Now run: python facedetection/train_face_recognition.py")
        print("   to train the face recognition model!\n")


if __name__ == "__main__":
    capture_faces()
