"""
Understanding AdversarialPatchPyTorch parameters
"""

attack = AdversarialPatchPyTorch(
    estimator=classifier,                # REQUIRED: Your PyTorch classifier
    
    # Transformation parameters (for physical robustness):
    rotation_max=22.5,                   # Max rotation angle in degrees [0, 180]
    scale_min=0.1,                       # Min patch size (as fraction of image)
    scale_max=1.0,                       # Max patch size (as fraction of image)
    distortion_scale_max=0.0,            # Perspective distortion [0, 1], 0=disabled
    
    # Optimization parameters:
    learning_rate=5.0,                   # Learning rate (higher = faster but less stable)
    max_iter=500,                        # Number of training iterations
    batch_size=16,                       # Batch size for training
    optimizer='Adam',                    # 'Adam' or 'pgd'
    
    # Patch configuration:
    patch_shape=(3, 224, 224),          # (channels, height, width) of patch
    patch_location=None,                 # None=random, or (x, y) for fixed location
    patch_type='circle',                 # 'circle' or 'square'
    
    # Attack configuration:
    targeted=True,                       # True=targeted, False=untargeted
    verbose=True,                        # Show progress bar
    summary_writer=False,                # TensorBoard logging
)