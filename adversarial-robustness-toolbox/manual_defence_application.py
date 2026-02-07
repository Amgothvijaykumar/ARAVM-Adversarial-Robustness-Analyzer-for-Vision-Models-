"""
Manually applying defences to images (not integrated with classifier)
"""

# Method 1: Apply defence manually by calling it directly
spatial_smoothing = SpatialSmoothing(
    window_size=5,  # Larger window for more smoothing
    channels_first=True,
    clip_values=(0.0, 1.0)
)

# Apply to adversarial examples manually
x_test_adv_smoothed, _ = spatial_smoothing(x_test_adv, y_test[:1000])

print("\nManual Defence Application:")
print(f"Original shape: {x_test_adv.shape}")
print(f"After smoothing: {x_test_adv_smoothed.shape}")

# Get predictions on manually defended images
predictions_manual = classifier_no_defence.predict(x_test_adv_smoothed)
accuracy_manual = np.sum(
    np.argmax(predictions_manual, axis=1) == np.argmax(y_test[:1000], axis=1)
) / 1000

print(f"Adversarial Accuracy (manual defence): {accuracy_manual * 100:.2f}%")

# Method 2: Apply JPEG compression manually
jpeg_compression = JpegCompression(
    clip_values=(0.0, 1.0),
    quality=60,
    channels_first=True,
    verbose=False
)

x_test_adv_jpeg, _ = jpeg_compression(x_test_adv, y_test[:1000])

predictions_jpeg_manual = classifier_no_defence.predict(x_test_adv_jpeg)
accuracy_jpeg_manual = np.sum(
    np.argmax(predictions_jpeg_manual, axis=1) == np.argmax(y_test[:1000], axis=1)
) / 1000

print(f"Adversarial Accuracy (JPEG compression): {accuracy_jpeg_manual * 100:.2f}%")