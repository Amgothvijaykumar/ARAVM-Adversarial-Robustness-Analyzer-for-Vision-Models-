"""
Using ART's compute_success utilities for detailed attack analysis
"""

from art.utils import compute_success, compute_success_array

# Generate adversarial examples
attack = FastGradientMethod(estimator=classifier, eps=0.3)
x_test_adv = attack.generate(x=x_test)

# Method 1: Overall success rate
success_rate = compute_success(
    classifier=classifier,
    x_clean=x_test,
    labels=y_test,
    x_adv=x_test_adv,
    targeted=False,  # Untargeted attack
    batch_size=128
)

print(f"Overall Attack Success Rate: {success_rate * 100:.2f}%")

# Method 2: Per-sample success (returns boolean array)
success_array = compute_success_array(
    classifier=classifier,
    x_clean=x_test,
    labels=y_test,
    x_adv=x_test_adv,
    targeted=False,
    batch_size=128
)

print(f"Successful attacks: {np.sum(success_array)}/{len(success_array)}")
print(f"Failed attacks: {len(success_array) - np.sum(success_array)}/{len(success_array)}")

# Analyze which samples were successfully attacked
successful_indices = np.where(success_array)[0]
failed_indices = np.where(~success_array)[0]

print(f"\nFirst 10 successful attack indices: {successful_indices[:10]}")
print(f"First 10 failed attack indices: {failed_indices[:10]}")

# Calculate perturbation statistics for successful attacks
perturbations_successful = np.linalg.norm(
    (x_test_adv[successful_indices] - x_test[successful_indices]).reshape(len(successful_indices), -1),
    ord=2,
    axis=1
)

print(f"\nPerturbation statistics for successful attacks:")
print(f"  Mean: {np.mean(perturbations_successful):.6f}")
print(f"  Std:  {np.std(perturbations_successful):.6f}")
print(f"  Min:  {np.min(perturbations_successful):.6f}")
print(f"  Max:  {np.max(perturbations_successful):.6f}")