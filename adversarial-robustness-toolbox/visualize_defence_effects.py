"""
Visualize how defences modify adversarial examples
"""

import matplotlib.pyplot as plt

def visualize_defence_effects(x_clean, x_adv, defences, num_samples=5):
    """
    Visualize the effect of different defences on adversarial examples.
    """
    
    num_defences = len(defences) + 2  # clean, adv, + defences
    fig, axes = plt.subplots(num_samples, num_defences, figsize=(3 * num_defences, 3 * num_samples))
    
    for i in range(num_samples):
        # Column 0: Clean image
        img_clean = x_clean[i].squeeze()
        axes[i, 0].imshow(img_clean, cmap='gray')
        axes[i, 0].set_title('Clean' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Column 1: Adversarial image
        img_adv = x_adv[i].squeeze()
        axes[i, 1].imshow(img_adv, cmap='gray')
        axes[i, 1].set_title('Adversarial' if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Remaining columns: Apply each defence
        for j, (defence_name, defence) in enumerate(defences.items(), start=2):
            x_defended, _ = defence(x_adv[i:i+1], None)
            img_defended = x_defended[0].squeeze()
            
            axes[i, j].imshow(img_defended, cmap='gray')
            axes[i, j].set_title(defence_name if i == 0 else '')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('defence_effects_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# Create defences to visualize
defences_to_visualize = {
    'Spatial (3x3)': SpatialSmoothing(
        window_size=3,
        channels_first=True,
        clip_values=(0.0, 1.0)
    ),
    'Spatial (7x7)': SpatialSmoothing(
        window_size=7,
        channels_first=True,
        clip_values=(0.0, 1.0)
    ),
    'JPEG (Q=30)': JpegCompression(
        clip_values=(0.0, 1.0),
        quality=30,
        channels_first=True,
        verbose=False
    ),
    'JPEG (Q=75)': JpegCompression(
        clip_values=(0.0, 1.0),
        quality=75,
        channels_first=True,
        verbose=False
    ),
}

visualize_defence_effects(
    x_test[:5],
    x_test_adv[:5],
    defences_to_visualize,
    num_samples=5
)