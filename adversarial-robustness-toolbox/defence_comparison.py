"""
Comprehensive comparison of defence effectiveness across different attacks
"""

from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method
)

def evaluate_defence_effectiveness(
    classifier_no_def,
    classifier_with_def,
    x_test,
    y_test,
    attacks,
    defence_name
):
    """
    Evaluate how well a defence protects against various attacks.
    """
    
    results = {
        'attack': [],
        'clean_acc': [],
        'adv_acc_no_def': [],
        'adv_acc_with_def': [],
        'improvement': []
    }
    
    # Clean accuracy
    predictions_clean = classifier_no_def.predict(x_test)
    clean_acc = np.sum(
        np.argmax(predictions_clean, axis=1) == np.argmax(y_test, axis=1)
    ) / len(y_test)
    
    print(f"\n{'='*70}")
    print(f"DEFENCE EVALUATION: {defence_name}")
    print(f"{'='*70}")
    print(f"Clean Accuracy: {clean_acc * 100:.2f}%\n")
    
    for attack_name, attack in attacks.items():
        print(f"Testing against {attack_name}...")
        
        # Generate adversarial examples
        x_test_adv = attack.generate(x=x_test)
        
        # Evaluate without defence
        predictions_no_def = classifier_no_def.predict(x_test_adv)
        acc_no_def = np.sum(
            np.argmax(predictions_no_def, axis=1) == np.argmax(y_test, axis=1)
        ) / len(y_test)
        
        # Evaluate with defence
        predictions_with_def = classifier_with_def.predict(x_test_adv)
        acc_with_def = np.sum(
            np.argmax(predictions_with_def, axis=1) == np.argmax(y_test, axis=1)
        ) / len(y_test)
        
        improvement = acc_with_def - acc_no_def
        
        results['attack'].append(attack_name)
        results['clean_acc'].append(clean_acc)
        results['adv_acc_no_def'].append(acc_no_def)
        results['adv_acc_with_def'].append(acc_with_def)
        results['improvement'].append(improvement)
        
        print(f"  No Defence:   {acc_no_def * 100:5.2f}%")
        print(f"  With Defence: {acc_with_def * 100:5.2f}%")
        print(f"  Improvement:  {improvement * 100:+5.2f}%\n")
    
    return results

# Create attacks to test
attacks = {
    'FGSM (ε=0.1)': FastGradientMethod(
        estimator=classifier_no_defence,
        eps=0.1
    ),
    'FGSM (ε=0.3)': FastGradientMethod(
        estimator=classifier_no_defence,
        eps=0.3
    ),
    'PGD (ε=0.3)': ProjectedGradientDescent(
        estimator=classifier_no_defence,
        eps=0.3,
        eps_step=0.01,
        max_iter=40
    ),
}

# Test SpatialSmoothing
results_spatial = evaluate_defence_effectiveness(
    classifier_no_defence,
    classifier_with_defence,
    x_test[:500],
    y_test[:500],
    attacks,
    "Spatial Smoothing (window_size=3)"
)

# Test JPEG Compression
results_jpeg = evaluate_defence_effectiveness(
    classifier_no_defence,
    classifier_jpeg,
    x_test[:500],
    y_test[:500],
    attacks,
    "JPEG Compression (quality=50)"
)

# Test Combined Defences
results_combined = evaluate_defence_effectiveness(
    classifier_no_defence,
    classifier_combined,
    x_test[:500],
    y_test[:500],
    attacks,
    "Combined Defences"
)