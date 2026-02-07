"""
This example demonstrates how epsilon affects attack strength.
Lower epsilon = smaller perturbations (harder to detect, potentially less effective)
Higher epsilon = larger perturbations (easier to detect, more effective)
"""

import numpy as np
from art.attacks.evasion import FastGradientMethod

# Assuming you have a trained classifier and test data
# classifier = ... (your trained PyTorchClassifier or KerasClassifier)
# x_test, y_test = ... (your test data)

# Define different epsilon values to test
epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

print("\nComparing attack effectiveness with different epsilon values:")
print("-" * 60)

# Iterate over epsilon values
for epsilon in epsilon_values:
    # Create FGSM attack with specific epsilon
    attack = FastGradientMethod(classifier, eps=epsilon)
    
    # Generate adversarial examples
    x_test_adv = attack.generate(x=x_test, y=y_test)
    
    # Evaluate the classifier on adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    accuracy = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    
    print("Epsilon: {:.2f} | Accuracy: {:.2f}%".format(epsilon, accuracy * 100))

print("-" * 60)