"""
Understanding defence limitations and best practices

Important papers on defence limitations:
- https://arxiv.org/abs/1803.09868 (SpatialSmoothing limitations)
- https://arxiv.org/abs/1802.00420 (JPEG compression limitations)
- https://arxiv.org/abs/1902.06705 (General evaluation guidelines)
"""

print("""
DEFENCE BEST PRACTICES:

1. SpatialSmoothing:
   ✓ Effective against high-frequency perturbations (FGSM, PGD)
   ✓ Low computational cost
   ✓ Can be used in real-time applications
   ✗ May reduce clean accuracy by blurring images
   ✗ Vulnerable to adaptive attacks
   ✗ Less effective against large perturbations
   
   Recommendations:
   - Use window_size=3 or 5 (balance between defence and accuracy)
   - Best for images with low-frequency content
   - Combine with other defences for better protection

2. JPEG Compression:
   ✓ Removes high-frequency noise effectively
   ✓ Simple to implement and understand
   ✓ Works well against gradient-based attacks
   ✗ Significantly degrades image quality at low quality settings
   ✗ May hurt clean accuracy
   ✗ Can be bypassed by adaptive attacks
   
   Recommendations:
   - Use quality=50-75 for balance
   - Quality < 50 causes significant quality loss
   - Quality > 80 provides minimal defence
   - Best for RGB images (3 channels)

3. Combined Defences:
   ✓ Multiple layers of protection
   ✓ Harder for attackers to bypass all defences
   ✗ Cumulative accuracy drop on clean data
   ✗ Increased computational cost
   
   Recommendations:
   - Order matters: apply less aggressive defences first
   - Monitor cumulative effect on clean accuracy
   - Test against adaptive attacks

4. General Guidelines:
   - Always evaluate on clean AND adversarial data
   - Test against multiple attack types
   - Be aware of adaptive attacks (attacks that know about your defence)
   - Consider the trade-off: defence strength vs clean accuracy
   - Defences are not foolproof - use as part of defense-in-depth
""")

# Example: Measure clean accuracy drop
def measure_clean_accuracy_drop(classifier_no_def, classifier_with_def, x_test, y_test):
    """
    Measure how much a defence hurts clean accuracy.
    """
    
    pred_no_def = classifier_no_def.predict(x_test)
    acc_no_def = np.sum(np.argmax(pred_no_def, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    
    pred_with_def = classifier_with_def.predict(x_test)
    acc_with_def = np.sum(np.argmax(pred_with_def, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    
    drop = acc_no_def - acc_with_def
    drop_percent = (drop / acc_no_def) * 100 if acc_no_def > 0 else 0
    
    print(f"\nClean Accuracy Impact:")
    print(f"  Without Defence: {acc_no_def * 100:.2f}%")
    print(f"  With Defence:    {acc_with_def * 100:.2f}%")
    print(f"  Drop:            {drop * 100:.2f}% ({drop_percent:.1f}% relative)")
    
    return drop

# Measure for different defences
print("\n" + "="*60)
print("CLEAN ACCURACY DROP ANALYSIS")
print("="*60)

drop_spatial = measure_clean_accuracy_drop(
    classifier_no_defence,
    classifier_with_defence,
    x_test[:1000],
    y_test[:1000]
)

drop_jpeg = measure_clean_accuracy_drop(
    classifier_no_defence,
    classifier_jpeg,
    x_test[:1000],
    y_test[:1000]
)

drop_combined = measure_clean_accuracy_drop(
    classifier_no_defence,
    classifier_combined,
    x_test[:1000],
    y_test[:1000]
)