"""
Example: Apply FGSM to a single test image with different epsilon values
"""

import numpy as np
from art.attacks.evasion import FastGradientMethod

def attack_single_image(classifier, image, true_label, epsilon):
    """
    Apply FGSM attack to a single image.
    
    :param classifier: Trained ART classifier
    :param image: Single image (add batch dimension if needed)
    :param true_label: True label for the image
    :param epsilon: Perturbation magnitude
    :return: Adversarial image, original prediction, adversarial prediction
    """
    
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Get original prediction
    orig_pred = classifier.predict(image)
    orig_class = np.argmax(orig_pred, axis=1)[0]
    
    # Create FGSM attack
    attack = FastGradientMethod(
        estimator=classifier,
        eps=epsilon,
        targeted=False
    )
    
    # Generate adversarial example
    # Note: Can pass y (label) or let it use model predictions
    adv_image = attack.generate(x=image)
    
    # Get adversarial prediction
    adv_pred = classifier.predict(adv_image)
    adv_class = np.argmax(adv_pred, axis=1)[0]
    
    # Calculate perturbation
    perturbation = adv_image - image
    perturbation_norm = np.linalg.norm(perturbation)
    
    print(f"Epsilon: {epsilon}")
    print(f"Original prediction: {orig_class} (confidence: {orig_pred[0, orig_class]:.4f})")
    print(f"Adversarial prediction: {adv_class} (confidence: {adv_pred[0, adv_class]:.4f})")
    print(f"True label: {np.argmax(true_label)}")
    print(f"Perturbation L2 norm: {perturbation_norm:.4f}")
    print(f"Attack successful: {orig_class != adv_class}\n")
    
    return adv_image[0], orig_class, adv_class

# Usage example
single_image = x_test[0]  # Select first test image
single_label = y_test[0]

# Test different epsilon values on the same image
print("Testing FGSM on a single image with different epsilon values:\n")
for eps in [0.05, 0.1, 0.2, 0.3]:
    adv_img, orig, adv = attack_single_image(classifier, single_image, single_label, eps)
    print("-" * 60)