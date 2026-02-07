"""
Visualize the adversarial patch and its effect
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize_patch_attack(attack, x_test, patch, patch_mask, classifier):
    """
    Visualize the adversarial patch and its effect on images.
    """
    
    # Apply patch to test images at different scales
    scales = [0.1, 0.3, 0.5, 0.7]
    
    fig, axes = plt.subplots(len(scales) + 1, 5, figsize=(15, 3 * (len(scales) + 1)))
    
    # Row 0: Show the patch itself
    patch_img = np.transpose(patch, (1, 2, 0))  # CHW -> HWC
    patch_img = np.clip(patch_img, 0, 1)
    
    axes[0, 0].imshow(patch_img)
    axes[0, 0].set_title('Adversarial Patch')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(patch_mask[0], cmap='gray')
    axes[0, 1].set_title('Patch Mask')
    axes[0, 1].axis('off')
    
    for i in range(2, 5):
        axes[0, i].axis('off')
    
    # Rows 1-4: Show patched images at different scales
    for row, scale in enumerate(scales, start=1):
        # Apply patch at this scale
        x_patched = attack.apply_patch(x=x_test[:3], scale=scale)
        
        for col in range(3):
            # Original image
            if row == 1:
                img_orig = np.transpose(x_test[col], (1, 2, 0))
                img_orig = np.clip(img_orig, 0, 1)
                axes[0, col + 2].imshow(img_orig)
                axes[0, col + 2].set_title(f'Original {col + 1}')
                axes[0, col + 2].axis('off')
            
            # Patched image
            img_patched = np.transpose(x_patched[col], (1, 2, 0))
            img_patched = np.clip(img_patched, 0, 1)
            
            # Get predictions
            pred_orig = np.argmax(classifier.predict(x_test[col:col+1]), axis=1)[0]
            pred_patch = np.argmax(classifier.predict(x_patched[col:col+1]), axis=1)[0]
            
            axes[row, col].imshow(img_patched)
            axes[row, col].set_title(f'Scale {scale}\nOrig: {pred_orig} → {pred_patch}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for col in range(3, 5):
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('adversarial_patch_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# Usage
visualize_patch_attack(attack, x_test, patch, patch_mask, classifier)