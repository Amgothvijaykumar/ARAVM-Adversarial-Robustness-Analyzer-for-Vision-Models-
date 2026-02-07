"""
Empirical Robustness: Measures the minimal perturbation needed to fool the classifier

Paper: https://arxiv.org/abs/1511.04599

This metric computes the average minimal perturbation required to change the model's prediction.
Higher values = more robust model
"""

from art.metrics import empirical_robustness

# Calculate empirical robustness using FGSM
emp_robust_fgsm = empirical_robustness(
    classifier=classifier,
    x=x_test[:500],  # Use smaller subset (computationally expensive)
    attack_name='fgsm',
    attack_params={'eps': 1.0, 'eps_step': 0.01}  # eps is max perturbation
)

print(f"\nEmpirical Robustness (FGSM): {emp_robust_fgsm:.4f}")

# Calculate empirical robustness using HopSkipJump (more accurate but slower)
emp_robust_hsj = empirical_robustness(
    classifier=classifier,
    x=x_test[:100],  # Use even smaller subset
    attack_name='hsj',
    attack_params={
        'max_iter': 50,
        'max_eval': 1000,
        'init_eval': 100
    }
)

print(f"Empirical Robustness (HSJ): {emp_robust_hsj:.4f}")

# Interpretation:
# - Empirical robustness of 0.1 means: on average, a perturbation of 10% of the input norm
#   is sufficient to fool the classifier
# - Higher values indicate more robust models