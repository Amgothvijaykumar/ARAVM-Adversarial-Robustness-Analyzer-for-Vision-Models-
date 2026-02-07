"""
Finding optimal defence parameters through grid search
"""

import matplotlib.pyplot as plt

def tune_spatial_smoothing_window(classifier_no_def, x_test, y_test, x_test_adv, window_sizes):
    """
    Find optimal window size for SpatialSmoothing defence.
    """
    
    results = {
        'window_size': [],
        'clean_acc': [],
        'adv_acc': []
    }
    
    print("\nTuning SpatialSmoothing window size:")
    print(f"{'Window':<10} {'Clean Acc':<12} {'Adv Acc':<12} {'Drop':<10}")
    print("-" * 50)
    
    for window_size in window_sizes:
        # Create defence with this window size
        defence = SpatialSmoothing(
            window_size=window_size,
            channels_first=True,
            clip_values=(0.0, 1.0),
            apply_predict=True
        )
        
        # Apply defence manually
        x_test_defended, _ = defence(x_test, None)
        x_test_adv_defended, _ = defence(x_test_adv, None)
        
        # Evaluate
        predictions_clean = classifier_no_def.predict(x_test_defended)
        clean_acc = np.sum(
            np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)
        ) / len(y_test)
        
        predictions_adv = classifier_no_def.predict(x_test_adv_defended)
        adv_acc = np.sum(
            np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)
        ) / len(y_test)
        
        drop = clean_acc - adv_acc
        
        results['window_size'].append(window_size)
        results['clean_acc'].append(clean_acc)
        results['adv_acc'].append(adv_acc)
        
        print(f"{window_size:<10} {clean_acc*100:<12.2f} {adv_acc*100:<12.2f} {drop*100:<10.2f}")
    
    return results

def tune_jpeg_quality(classifier_no_def, x_test, y_test, x_test_adv, quality_values):
    """
    Find optimal quality setting for JPEG compression defence.
    """
    
    results = {
        'quality': [],
        'clean_acc': [],
        'adv_acc': []
    }
    
    print("\nTuning JPEG Compression quality:")
    print(f"{'Quality':<10} {'Clean Acc':<12} {'Adv Acc':<12} {'Drop':<10}")
    print("-" * 50)
    
    for quality in quality_values:
        # Create defence with this quality
        defence = JpegCompression(
            clip_values=(0.0, 1.0),
            quality=quality,
            channels_first=True,
            apply_predict=True,
            verbose=False
        )
        
        # Apply defence manually
        x_test_defended, _ = defence(x_test, None)
        x_test_adv_defended, _ = defence(x_test_adv, None)
        
        # Evaluate
        predictions_clean = classifier_no_def.predict(x_test_defended)
        clean_acc = np.sum(
            np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)
        ) / len(y_test)
        
        predictions_adv = classifier_no_def.predict(x_test_adv_defended)
        adv_acc = np.sum(
            np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)
        ) / len(y_test)
        
        drop = clean_acc - adv_acc
        
        results['quality'].append(quality)
        results['clean_acc'].append(clean_acc)
        results['adv_acc'].append(adv_acc)
        
        print(f"{quality:<10} {clean_acc*100:<12.2f} {adv_acc*100:<12.2f} {drop*100:<10.2f}")
    
    return results

# Tune parameters
window_sizes = [3, 5, 7, 9]
results_window = tune_spatial_smoothing_window(
    classifier_no_defence,
    x_test[:200],
    y_test[:200],
    x_test_adv[:200],
    window_sizes
)

quality_values = [25, 50, 75, 90]
results_quality = tune_jpeg_quality(
    classifier_no_defence,
    x_test[:200],
    y_test[:200],
    x_test_adv[:200],
    quality_values
)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot window size tuning
axes[0].plot(results_window['window_size'], 
             np.array(results_window['clean_acc']) * 100, 
             'b-o', label='Clean Accuracy')
axes[0].plot(results_window['window_size'], 
             np.array(results_window['adv_acc']) * 100, 
             'r-o', label='Adversarial Accuracy')
axes[0].set_xlabel('Window Size')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Spatial Smoothing: Window Size')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot JPEG quality tuning
axes[1].plot(results_quality['quality'], 
             np.array(results_quality['clean_acc']) * 100, 
             'b-o', label='Clean Accuracy')
axes[1].plot(results_quality['quality'], 
             np.array(results_quality['adv_acc']) * 100, 
             'r-o', label='Adversarial Accuracy')
axes[1].set_xlabel('JPEG Quality')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('JPEG Compression: Quality')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('defence_parameter_tuning.png', dpi=150, bbox_inches='tight')
plt.show()