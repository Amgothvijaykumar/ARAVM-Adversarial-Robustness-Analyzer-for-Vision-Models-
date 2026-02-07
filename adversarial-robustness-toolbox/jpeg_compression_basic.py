"""
JPEG Compression: Removes adversarial perturbations through lossy compression
Papers: https://arxiv.org/abs/1705.02900, https://arxiv.org/abs/1608.00853

This defence applies JPEG compression which removes high-frequency noise
introduced by adversarial attacks.
"""

from art.defences.preprocessor import JpegCompression

# Create JPEG Compression defence
jpeg_compression = JpegCompression(
    clip_values=(0.0, 1.0),     # Data range
    quality=50,                  # JPEG quality (1-95, lower = more compression)
    channels_first=True,         # PyTorch format
    apply_fit=False,            # Don't apply during training
    apply_predict=True,         # Apply during prediction
    verbose=True                # Show progress bar
)

print("JPEG Compression defence created:")
print(f"  Quality: {jpeg_compression.quality}")
print(f"  Channels first: {jpeg_compression.channels_first}")

# Create classifier with JPEG compression defence
classifier_jpeg = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    preprocessing_defences=[jpeg_compression]
)

# Evaluate with JPEG compression
predictions_adv_jpeg = classifier_jpeg.predict(x_test_adv)
accuracy_adv_jpeg = np.sum(
    np.argmax(predictions_adv_jpeg, axis=1) == np.argmax(y_test[:1000], axis=1)
) / 1000

print(f"\nWith JPEG Compression Defence:")
print(f"  Adversarial Accuracy: {accuracy_adv_jpeg * 100:.2f}%")
print(f"  Improvement over no defence: {(accuracy_adv_jpeg - accuracy_adv_no_def) * 100:.2f}%")