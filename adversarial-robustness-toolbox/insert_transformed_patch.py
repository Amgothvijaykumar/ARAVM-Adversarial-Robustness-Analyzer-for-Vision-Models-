"""
Manually insert a patch with specific transformation coordinates
"""

# Define the 4 corners of where the patch should appear in the image
# Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
# Going clockwise from top-left
image_coords = np.array([
    [50, 50],    # Top-left corner
    [150, 60],   # Top-right corner (slightly rotated)
    [145, 160],  # Bottom-right corner
    [55, 150]    # Bottom-left corner
])

# Insert patch at these exact coordinates
x_image = x_test[0].copy()  # Single image
x_with_patch = AdversarialPatchPyTorch.insert_transformed_patch(
    x=x_image,
    patch=patch,
    image_coords=image_coords
)

print("Patch inserted at custom coordinates with perspective transformation!")