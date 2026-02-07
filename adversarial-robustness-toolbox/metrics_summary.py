"""
Summary of available metrics in ART
"""

# 1. ACCURACY METRICS
# -------------------
# Clean Accuracy: Performance on unmodified test data
clean_accuracy = np.sum(np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

# Adversarial Accuracy: Performance on adversarial examples
# Using art.metrics.adversarial_accuracy
from art.metrics import adversarial_accuracy
adv_acc = adversarial_accuracy(classifier, x_test, y_test, attack_name='fgsm', attack_params={'eps': 0.3})

# 2. ROBUSTNESS METRICS
# ---------------------
# Empirical Robustness: Minimal perturbation needed to fool the model
# Using art.metrics.empirical_robustness
from art.metrics import empirical_robustness
emp_robust = empirical_robustness(classifier, x_test, attack_name='fgsm', attack_params={'eps': 1.0})

# 3. ATTACK SUCCESS METRICS
# -------------------------
# Attack Success Rate: Percentage of successful adversarial examples
# Using art.utils.compute_success
from art.utils import compute_success
success_rate = compute_success(classifier, x_test, y_test, x_test_adv, targeted=False)

# 4. PERTURBATION METRICS
# -----------------------
# L0, L1, L2, L∞ norms of perturbations
perturbation_l2 = np.mean(np.linalg.norm((x_adv - x_clean).reshape(n, -1), ord=2, axis=1))
perturbation_linf = np.mean(np.linalg.norm((x_adv - x_clean).reshape(n, -1), ord=np.inf, axis=1))