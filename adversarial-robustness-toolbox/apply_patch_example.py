"""
Apply a trained patch to new images
"""

import matplotlib.pyplot as plt

# Load test images
x_test = np.random.rand(10, 3, 224, 224).astype(np.float32)

# Method 1: Apply the learned patch with random transformations
scale = 0.3  # Patch will be 30% of image size
x_test_patched = attack.apply_patch(
    x=x_test,
    scale=scale
)

# Method 2: Apply an external patch (e.g., previously saved)
external_patch = patch  # Use the patch we just generated
x_test_patched_external = attack.apply_patch(
    x=x_test,
    scale=0.4,
    patch_external=external_patch
)

# Method 3: Apply patch with location mask
# Create a mask specifying where the patch center can be placed
mask = np.ones((10, 224, 224), dtype=bool)
# Only allow patch in the upper-left quadrant
mask[:, 112:, :] = False  # Block bottom half
mask[:, :, 112:] = False  # Block right half

x_test_masked = attack.apply_patch(
    x=x_test,
    scale=0.3,
    mask=mask
)

print("Patched images created!")