"""
Generate a comprehensive robustness report
"""

def generate_robustness_report(classifier, x_test, y_test, attacks, output_file='robustness_report.txt'):
    """
    Generate a comprehensive robustness evaluation report.
    """
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" " * 15 + "MODEL ROBUSTNESS EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # 1. Clean Performance
        f.write("1. BASELINE PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        
        predictions_clean = classifier.predict(x_test)
        clean_acc = np.sum(np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        
        f.write(f"   Test Samples: {len(y_test)}\n")
        f.write(f"   Clean Accuracy: {clean_acc * 100:.2f}%\n\n")
        
        # 2. Attack Results
        f.write("2. ADVERSARIAL ATTACK RESULTS\n")
        f.write("-" * 70 + "\n\n")
        
        for attack_name, attack in attacks.items():
            f.write(f"   Attack: {attack_name}\n")
            f.write(f"   {'-' * 65}\n")
            
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
                targeted=False
            )
            
            perturbation_l2 = np.mean(
                np.linalg.norm((x_test_adv - x_test).reshape(x_test.shape[0], -1), ord=2, axis=1)
            )
            perturbation_linf = np.mean(
                np.linalg.norm((x_test_adv - x_test).reshape(x_test.shape[0], -1), ord=np.inf, axis=1)
            )
            
            f.write(f"   Adversarial Accuracy: {adv_acc * 100:.2f}%\n")
            f.write(f"   Attack Success Rate: {success_rate * 100:.2f}%\n")
            f.write(f"   Accuracy Drop: {(clean_acc - adv_acc) * 100:.2f}%\n")
            f.write(f"   Average L2 Perturbation: {perturbation_l2:.6f}\n")
            f.write(f"   Average L∞ Perturbation: {perturbation_linf:.6f}\n\n")
        
        # 3. Robustness Summary
        f.write("3. ROBUSTNESS SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Model shows varying robustness across different attacks.\n")
        f.write(f"   Clean accuracy: {clean_acc * 100:.2f}%\n")
        f.write(f"   Evaluated {len(attacks)} different attack scenarios.\n\n")
        
        f.write("="*70 + "\n")
        f.write("Report generated successfully.\n")
        f.write("="*70 + "\n")
    
    print(f"Report saved to: {output_file}")

# Usage
attacks = {
    'FGSM (ε=0.1)': FastGradientMethod(estimator=classifier, eps=0.1),
    'FGSM (ε=0.3)': FastGradientMethod(estimator=classifier, eps=0.3),
    'PGD (ε=0.3)': ProjectedGradientDescent(
        estimator=classifier,
        eps=0.3,
        eps_step=0.01,
        max_iter=40
    )
}

generate_robustness_report(
    classifier=classifier,
    x_test=x_test[:1000],
    y_test=y_test[:1000],
    attacks=attacks,
    output_file='model_robustness_report.txt'
)