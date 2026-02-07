"""
Enable perspective distortion for more robust physical-world attacks
"""

attack_distorted = AdversarialPatchPyTorch(
    estimator=classifier,
    rotation_max=45.0,           # Allow more rotation
    scale_min=0.2,
    scale_max=0.8,
    distortion_scale_max=0.2,    # Enable perspective distortion (0-20% distortion)
    learning_rate=5.0,
    max_iter=1000,               # More iterations for better convergence
    batch_size=16,
    patch_shape=(3, 100, 100),
    patch_type='square',
    optimizer='Adam',
    targeted=True,
    verbose=True
)

# This creates patches that are more robust to:
# - Camera viewing angles
# - Non-planar surfaces
# - Real-world perspective changes

patch_robust, mask_robust = attack_distorted.generate(x=x_train, y=y_target)