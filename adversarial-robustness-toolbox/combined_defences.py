"""
Combining multiple preprocessing defences for stronger protection
"""

from art.defences.preprocessor import (
    SpatialSmoothing,
    JpegCompression,
    FeatureSqueezing,
    GaussianAugmentation
)

# Create multiple defences
spatial_smoothing = SpatialSmoothing(
    window_size=3,
    channels_first=True,
    clip_values=(0.0, 1.0),
    apply_predict=True
)

jpeg_compression = JpegCompression(
    clip_values=(0.0, 1.0),
    quality=75,  # Higher quality preserves more details
    channels_first=True,
    apply_predict=True
)

feature_squeezing = FeatureSqueezing(
    bit_depth=4,  # Reduce color depth to remove noise
    clip_values=(0.0, 1.0),
    apply_predict=True
)

# Create classifier with multiple defences (applied in order)
classifier_combined = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    preprocessing_defences=[
        feature_squeezing,   # Applied first
        jpeg_compression,    # Applied second
        spatial_smoothing    # Applied last
    ]
)

print("\nCombined Defences:")
print(f"  1. Feature Squeezing (bit_depth={feature_squeezing.bit_depth})")
print(f"  2. JPEG Compression (quality={jpeg_compression.quality})")
print(f"  3. Spatial Smoothing (window_size={spatial_smoothing.window_size})")

# Evaluate combined defences
predictions_adv_combined = classifier_combined.predict(x_test_adv)
accuracy_adv_combined = np.sum(
    np.argmax(predictions_adv_combined, axis=1) == np.argmax(y_test[:1000], axis=1)
) / 1000

print(f"\nWith Combined Defences:")
print(f"  Adversarial Accuracy: {accuracy_adv_combined * 100:.2f}%")
print(f"  Improvement: {(accuracy_adv_combined - accuracy_adv_no_def) * 100:.2f}%")