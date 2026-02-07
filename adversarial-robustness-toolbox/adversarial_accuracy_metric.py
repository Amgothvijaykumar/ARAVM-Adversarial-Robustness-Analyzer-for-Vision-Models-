"""
Using ART's built-in adversarial_accuracy metric
"""

from art.metrics import adversarial_accuracy

# Method 1: Using attack name (preset attacks)
# Available: 'fgsm', 'auto', 'hsj'
adv_acc_fgsm = adversarial_accuracy(
    classifier=classifier,
    x=x_test[:1000],  # Use subset for faster computation
    y=y_test[:1000],
    attack_name='fgsm',
    attack_params={'eps': 0.3, 'eps_step': 0.1}
)

print(f"\nAdversarial Accuracy (FGSM, eps=0.3): {adv_acc_fgsm * 100:.2f}%")

# Method 2: Using custom attack instance
from art.attacks.evasion import ProjectedGradientDescent

custom_attack = ProjectedGradientDescent(
    estimator=classifier,
    eps=0.3,
    eps_step=0.01,
    max_iter=40,
    targeted=False,
)

adv_acc_pgd = adversarial_accuracy(
    classifier=classifier,
    x=x_test[:1000],
    y=y_test[:1000],
    attack_crafter=custom_attack
)

print(f"Adversarial Accuracy (PGD, eps=0.3): {adv_acc_pgd * 100:.2f}%")

# Method 3: Without labels (doesn't exclude wrong predictions)
adv_acc_no_labels = adversarial_accuracy(
    classifier=classifier,
    x=x_test[:1000],
    y=None,  # No labels provided
    attack_name='fgsm',
    attack_params={'eps': 0.2}
)

print(f"Adversarial Accuracy without labels: {adv_acc_no_labels * 100:.2f}%")