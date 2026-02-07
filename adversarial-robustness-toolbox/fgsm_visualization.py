"""
Visualize how different epsilon values affect the adversarial perturbations
"""

import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod

def visualize_fgsm_perturbations(classifier, x_test, y_test, epsilon_values, num_samples=5):
    """
    Visualize adversarial examples at different epsilon values.
    
    :param classifier: Trained ART classifier
    :param x_test: Test images
    :param y_test: Test labels
    :param epsilon_values: List of epsilon values to compare
    :param num_samples: Number of samples to visualize
    """
    
    # Select a few random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    x_samples = x_test[indices]
    y_samples = y_test[indices]
    
    # Get original predictions
    orig_preds = np.argmax(classifier.predict(x_samples), axis=1)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, len(epsilon_values) + 2, 
                             figsize=(3 * (len(epsilon_values) + 2), 3 * num_samples))
    
    for i in range(num_samples):
        # Display original image
        axes[i, 0].imshow(x_samples[i].squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original\nPred: {orig_preds[i]}')
        axes[i, 0].axis('off')
        
        # Display adversarial examples for each epsilon
        for j, eps in enumerate(epsilon_values):
            # Generate adversarial example
            attack = FastGradientMethod(classifier, eps=eps)
            x_adv = attack.generate(x=x_samples[i:i+1])
            
            # Get adversarial prediction
            adv_pred = np.argmax(classifier.predict(x_adv), axis=1)[0]
            
            # Calculate perturbation
            perturbation = x_adv[0] - x_samples[i]
            
            # Display adversarial image
            axes[i, j+1].imshow(x_adv[0].squeeze(), cmap='gray')
            axes[i, j+1].set_title(f'ε={eps}\nPred: {adv_pred}')
            axes[i, j+1].axis('off')
            
            # Display perturbation in last column if it's the last epsilon
            if j == len(epsilon_values) - 1:
                axes[i, -1].imshow(perturbation.squeeze(), cmap='seismic', vmin=-eps, vmax=eps)
                axes[i, -1].set_title('Perturbation')
                axes[i, -1].axis('off')
    
    plt.tight_layout()
    plt.savefig('fgsm_epsilon_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Usage
epsilon_values = [0.05, 0.1, 0.2, 0.3]
visualize_fgsm_perturbations(classifier, x_test, y_test, epsilon_values, num_samples=3)