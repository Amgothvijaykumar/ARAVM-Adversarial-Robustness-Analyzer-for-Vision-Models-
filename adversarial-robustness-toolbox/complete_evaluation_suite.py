"""
Complete evaluation comparing model performance before and after attack
"""

import numpy as np
from art.metrics import adversarial_accuracy, empirical_robustness
from art.utils import compute_success, compute_success_array
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

def evaluate_model_robustness(classifier, x_test, y_test, attack_configs):
    """
    Comprehensive robustness evaluation of a classifier.
    
    :param classifier: Trained ART classifier
    :param x_test: Test samples
    :param y_test: Test labels
    :param attack_configs: Dictionary of attack configurations
    :return: Dictionary with all metrics
    """
    
    results = {}
    
    # 1. Clean accuracy (baseline)
    predictions_clean = classifier.predict(x_test)
    clean_accuracy = np.sum(
        np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)
    ) / len(y_test)
    
    results['clean_accuracy'] = clean_accuracy
    print(f"\n{'='*60}")
    print(f"BASELINE METRICS")
    print(f"{'='*60}")
    print(f"Clean Accuracy: {clean_accuracy * 100:.2f}%")
    
    # 2. Evaluate each attack
    for attack_name, config in attack_configs.items():
        print(f"\n{'='*60}")
        print(f"ATTACK: {attack_name.upper()}")
        print(f"{'='*60}")
        
        attack = config['attack']
        
        # Generate adversarial examples
        x_test_adv = attack.generate(x=x_test)
        
        # Calculate adversarial accuracy
        predictions_adv = classifier.predict(x_test_adv)
        adv_accuracy = np.sum(
            np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)
        ) / len(y_test)
        
        # Calculate attack success rate
        success_rate = compute_success(
            classifier=classifier,
            x_clean=x_test,
            labels=y_test,
            x_adv=x_test_adv,
            targeted=False,
            batch_size=128
        )
        
        # Calculate per-sample success
        success_array = compute_success_array(
            classifier=classifier,
            x_clean=x_test,
            labels=y_test,
            x_adv=x_test_adv,
            targeted=False,
            batch_size=128
        )
        
        # Calculate average perturbation
        perturbation = np.mean(np.abs(x_test_adv - x_test))
        perturbation_l2 = np.mean(
            np.linalg.norm((x_test_adv - x_test).reshape(x_test.shape[0], -1), ord=2, axis=1)
        )
        perturbation_linf = np.mean(
            np.linalg.norm((x_test_adv - x_test).reshape(x_test.shape[0], -1), ord=np.inf, axis=1)
        )
        
        # Store results
        results[attack_name] = {
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': success_rate,
            'accuracy_drop': clean_accuracy - adv_accuracy,
            'average_perturbation': perturbation,
            'average_l2_perturbation': perturbation_l2,
            'average_linf_perturbation': perturbation_linf,
            'successful_attacks': np.sum(success_array),
            'total_samples': len(success_array)
        }
        
        # Print results
        print(f"Adversarial Accuracy: {adv_accuracy * 100:.2f}%")
        print(f"Attack Success Rate: {success_rate * 100:.2f}%")
        print(f"Accuracy Drop: {(clean_accuracy - adv_accuracy) * 100:.2f}%")
        print(f"Average Perturbation (L1): {perturbation:.6f}")
        print(f"Average Perturbation (L2): {perturbation_l2:.6f}")
        print(f"Average Perturbation (L∞): {perturbation_linf:.6f}")
        print(f"Successful Attacks: {np.sum(success_array)}/{len(success_array)}")
    
    # 3. Empirical Robustness (if requested)
    if 'compute_emp_robust' in attack_configs and attack_configs['compute_emp_robust']:
        print(f"\n{'='*60}")
        print(f"EMPIRICAL ROBUSTNESS")
        print(f"{'='*60}")
        
        emp_robust = empirical_robustness(
            classifier=classifier,
            x=x_test[:500],  # Use subset
            attack_name='fgsm',
            attack_params={'eps': 1.0, 'eps_step': 0.01}
        )
        
        results['empirical_robustness'] = emp_robust
        print(f"Empirical Robustness: {emp_robust:.6f}")
    
    return results

# Usage example
attack_configs = {
    'fgsm_0.1': {
        'attack': FastGradientMethod(estimator=classifier, eps=0.1)
    },
    'fgsm_0.3': {
        'attack': FastGradientMethod(estimator=classifier, eps=0.3)
    },
    'pgd_0.3': {
        'attack': ProjectedGradientDescent(
            estimator=classifier,
            eps=0.3,
            eps_step=0.01,
            max_iter=40,
            targeted=False
        )
    },
    'compute_emp_robust': True
}

# Run evaluation
results = evaluate_model_robustness(
    classifier=classifier,
    x_test=x_test[:1000],
    y_test=y_test[:1000],
    attack_configs=attack_configs
)