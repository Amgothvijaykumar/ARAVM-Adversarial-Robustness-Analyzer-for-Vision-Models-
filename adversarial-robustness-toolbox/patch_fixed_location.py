"""
Example: Create a patch at a specific location (e.g., top-left corner)
"""

from art.attacks.evasion.adversarial_patch import AdversarialPatchPyTorch

# Create attack with fixed patch location
attack_fixed = AdversarialPatchPyTorch(
    estimator=classifier,
    rotation_max=0.0,           # No rotation for fixed location
    scale_min=1.0,              # Fixed scale
    scale_max=1.0,              # Fixed scale
    learning_rate=5.0,
    max_iter=500,
    batch_size=16,
    patch_shape=(3, 50, 50),    # Smaller 50x50 patch
    patch_location=(10, 10),    # Place patch at coordinates (x=10, y=10)
    patch_type='square',        # Square patch
    targeted=True,
    verbose=True
)

# Generate patch at fixed location
patch_fixed, mask_fixed = attack_fixed.generate(x=x_train, y=y_target)

print(f"Fixed location patch created at position (10, 10)")