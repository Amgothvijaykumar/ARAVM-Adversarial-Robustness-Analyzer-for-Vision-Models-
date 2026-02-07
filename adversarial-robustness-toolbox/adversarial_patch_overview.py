"""
AdversarialPatchPyTorch - Physical-world adversarial patch attack

Paper: https://arxiv.org/abs/1712.09665

This attack generates an adversarial patch that can be printed and placed on objects
to fool image classifiers and object detectors in the physical world.
"""

from art.attacks.evasion.adversarial_patch import AdversarialPatchPyTorch
import numpy as np

# The patch is optimized through:
# 1. Random transformations (rotation, scaling, translation, distortion)
# 2. Applied to training images
# 3. Gradient descent to maximize misclassification