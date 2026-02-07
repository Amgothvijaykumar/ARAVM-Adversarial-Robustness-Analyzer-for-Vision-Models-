"""
FastGradientMethod supports several parameters for customization:
"""

from art.attacks.evasion import FastGradientMethod
import numpy as np

# Basic FGSM Attack (untargeted)
attack_basic = FastGradientMethod(
    estimator=classifier,
    eps=0.3,                # Maximum perturbation (default: 0.3)
    norm=np.inf,            # Norm for perturbation: np.inf, 1, 2, etc. (default: np.inf)
    batch_size=32,          # Batch size for generating adversarial examples (default: 32)
    minimal=False,          # Whether to find minimal perturbation (default: False)
)

# FGSM with L2 norm constraint
attack_l2 = FastGradientMethod(
    estimator=classifier,
    eps=2.0,
    norm=2,                 # Use L2 norm instead of L-infinity
    batch_size=128,
)

# Targeted FGSM Attack
attack_targeted = FastGradientMethod(
    estimator=classifier,
    eps=0.3,
    targeted=True,          # Make it a targeted attack
    batch_size=32,
)

# FGSM with random initialization (more robust attack)
attack_random = FastGradientMethod(
    estimator=classifier,
    eps=0.3,
    num_random_init=5,      # Number of random restarts (default: 0)
    batch_size=32,
)

# Minimal perturbation FGSM (finds smallest perturbation needed)
attack_minimal = FastGradientMethod(
    estimator=classifier,
    eps=1.0,                # Maximum allowed perturbation
    eps_step=0.01,          # Step size for minimal perturbation search
    minimal=True,           # Enable minimal perturbation mode
    batch_size=32,
)

# Generate adversarial examples
x_adv = attack_basic.generate(x=x_test)

# For targeted attacks, specify target labels
target_labels = np.zeros_like(y_test)
target_labels[:, 0] = 1  # Target class 0 for all samples
x_adv_targeted = attack_targeted.generate(x=x_test, y=target_labels)