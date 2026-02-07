"""
Tips for creating patches that work in the physical world
"""

# 1. Use larger patch_shape for printable patches
physical_patch_attack = AdversarialPatchPyTorch(
    estimator=classifier,
    patch_shape=(3, 300, 300),  # Larger patch for better print quality
    patch_type='square',         # Easier to print and mount
    rotation_max=45.0,          # Account for various viewing angles
    scale_min=0.15,
    scale_max=0.40,
    distortion_scale_max=0.15,  # Account for perspective
    max_iter=1000,              # More iterations for robustness
    targeted=True
)

# 2. Train on diverse data
# Include various:
# - Lighting conditions
# - Backgrounds
# - Distances
# - Viewing angles

# 3. Post-process for printing
patch_printable = np.clip(patch * 255, 0, 255).astype(np.uint8)
# Convert to PIL Image for printing
from PIL import Image
patch_img = Image.fromarray(np.transpose(patch_printable, (1, 2, 0)))
patch_img.save('adversarial_patch_to_print.png')

print("Patch saved for physical printing!")