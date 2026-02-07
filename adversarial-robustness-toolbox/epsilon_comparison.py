"""
Compare model robustness across different attack intensities
"""

import matplotlib.pyplot as plt

def evaluate_robustness_vs_epsilon(classifier, x_test, y_test, epsilon_values):
    """
    Evaluate how model robustness changes with attack strength.
    """
    
    results = {
        'epsilon': [],
        'clean_accuracy': [],
        'adversarial_accuracy': [],
        'attack_success_rate': [],
        'average_perturbation': []
    }
    
    # Get clean accuracy (constant across epsilons)
    predictions_clean = classifier.predict(x_test)
    clean_acc = np.sum(np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    
    print(f"Clean Accuracy: {clean_acc * 100:.2f}%\n")
    print(f"{'Epsilon':<10} {'Adv Acc':<12} {'Success':<12} {'Avg Pert':<12}")
    print(f"{'-'*50}")
    
    for eps in epsilon_values:
        # Create attack
        attack = FastGradientMethod(estimator=classifier, eps=eps)
        
        # Generate adversarial examples
        x_test_adv = attack.generate(x=x_test)
        
        # Calculate metrics
        predictions_adv = classifier.predict(x_test_adv)
        adv_acc = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        
        success_rate = compute_success(
            classifier=classifier,
            x_clean=x_test,
            labels=y_test,
            x_adv=x_test_adv,
            targeted=False,
            batch_size=128
        )
        
        avg_pert = np.mean(np.abs(x_test_adv - x_test))
        
        # Store results
        results['epsilon'].append(eps)
        results['clean_accuracy'].append(clean_acc)
        results['adversarial_accuracy'].append(adv_acc)
        results['attack_success_rate'].append(success_rate)
        results['average_perturbation'].append(avg_pert)
        
        print(f"{eps:<10.3f} {adv_acc*100:<12.2f} {success_rate*100:<12.2f} {avg_pert:<12.6f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Accuracy vs Epsilon
    axes[0].plot(results['epsilon'], np.array(results['clean_accuracy']) * 100, 
                 'b-', label='Clean Accuracy', linewidth=2)
    axes[0].plot(results['epsilon'], np.array(results['adversarial_accuracy']) * 100, 
                 'r-', label='Adversarial Accuracy', linewidth=2)
    axes[0].set_xlabel('Epsilon')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy vs Attack Strength')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Attack Success Rate vs Epsilon
    axes[1].plot(results['epsilon'], np.array(results['attack_success_rate']) * 100, 
                 'g-', linewidth=2)
    axes[1].set_xlabel('Epsilon')
    axes[1].set_ylabel('Attack Success Rate (%)')
    axes[1].set_title('Attack Success vs Strength')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Perturbation vs Epsilon
    axes[2].plot(results['epsilon'], results['average_perturbation'], 
                 'm-', linewidth=2)
    axes[2].set_xlabel('Epsilon')
    axes[2].set_ylabel('Average Perturbation')
    axes[2].set_title('Perturbation vs Epsilon')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('robustness_vs_epsilon.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results

# Run evaluation
epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
results = evaluate_robustness_vs_epsilon(
    classifier=classifier,
    x_test=x_test[:1000],
    y_test=y_test[:1000],
    epsilon_values=epsilon_values
)