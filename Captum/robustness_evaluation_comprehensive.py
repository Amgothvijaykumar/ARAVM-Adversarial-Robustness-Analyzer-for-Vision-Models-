import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from PIL import Image
from typing import Tuple, Callable, Dict, List

from captum.attr import (
    IntegratedGradients,
    Saliency,
    GradientShap,
    DeepLift,
    InputXGradient
)
from captum.metrics import infidelity, sensitivity_max

class ModelRobustnessEvaluator:
    """
    Comprehensive model robustness evaluation using Captum metrics
    Identifies security vulnerabilities in ML deployments
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(device)
        
    def evaluate_explanation_quality(self,
                                     inputs: torch.Tensor,
                                     attribution_methods: Dict[str, Callable],
                                     target: int = None,
                                     n_samples: int = 10) -> Dict:
        """
        Evaluate explanation quality using infidelity and sensitivity metrics
        
        Returns:
            Dictionary containing infidelity and sensitivity scores for each method
        """
        results = {}
        
        for method_name, attr_method in attribution_methods.items():
            print(f"\nEvaluating {method_name}...")
            
            # Compute attributions
            attributions = attr_method.attribute(inputs, target=target)
            
            # 1. INFIDELITY: Measures explanation faithfulness
            infidelity_score = self._compute_infidelity(
                inputs, 
                attributions, 
                target, 
                n_samples
            )
            
            # 2. SENSITIVITY: Measures explanation stability
            sensitivity_score = self._compute_sensitivity(
                attr_method,
                inputs,
                n_samples
            )
            
            results[method_name] = {
                'infidelity': infidelity_score.cpu().numpy(),
                'sensitivity': sensitivity_score.cpu().numpy(),
                'attributions': attributions
            }
            
            print(f"  Infidelity: {infidelity_score.mean():.6f} ± {infidelity_score.std():.6f}")
            print(f"  Sensitivity: {sensitivity_score.mean():.6f} ± {sensitivity_score.std():.6f}")
            
        return results
    
    def _compute_infidelity(self, 
                           inputs: torch.Tensor,
                           attributions: torch.Tensor,
                           target: int,
                           n_samples: int) -> torch.Tensor:
        """
        Compute infidelity score
        Lower is better (0 = perfect fidelity)
        """
        # Define perturbation function
        def perturb_fn(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns: (perturbations, perturbed_inputs)
            """
            # Add Gaussian noise
            noise_std = 0.1
            noise = torch.randn_like(inputs) * noise_std
            perturbed_inputs = inputs + noise
            
            # Clip to valid range
            perturbed_inputs = torch.clamp(perturbed_inputs, 
                                          inputs.min().item(), 
                                          inputs.max().item())
            
            # Perturbation magnitude
            perturbations = inputs - perturbed_inputs
            
            return perturbations, perturbed_inputs
        
        # Compute infidelity
        infid = infidelity(
            forward_func=self.model,
            perturb_func=perturb_fn,
            inputs=inputs,
            attributions=attributions,
            target=target,
            n_perturb_samples=n_samples,
            max_examples_per_batch=8
        )
        
        return infid
    
    def _compute_sensitivity(self,
                           explanation_func: Callable,
                           inputs: torch.Tensor,
                           n_samples: int) -> torch.Tensor:
        """
        Compute sensitivity max score
        Lower is better (stable explanations)
        """
        # Custom perturbation function for sensitivity
        def perturb_fn(inputs: torch.Tensor, 
                      perturb_radius: float = 0.02) -> torch.Tensor:
            """
            Perturb inputs within L-infinity ball
            """
            noise = torch.FloatTensor(inputs.shape).uniform_(
                -perturb_radius, perturb_radius
            ).to(inputs.device)
            
            perturbed = inputs + noise
            return torch.clamp(perturbed, inputs.min().item(), inputs.max().item())
        
        # Compute sensitivity
        sens = sensitivity_max(
            explanation_func=explanation_func.attribute,
            inputs=inputs,
            perturb_func=perturb_fn,
            perturb_radius=0.02,  # 2% perturbation radius
            n_perturb_samples=n_samples,
            max_examples_per_batch=8
        )
        
        return sens
    
    def test_adversarial_robustness(self,
                                   inputs: torch.Tensor,
                                   attr_method: Callable,
                                   target: int,
                                   epsilon_values: List[float] = None) -> Dict:
        """
        Test how explanations change under adversarial perturbations
        
        This identifies security vulnerabilities:
        - High sensitivity = easy to fool explanations
        - Unstable explanations = unreliable in adversarial settings
        """
        if epsilon_values is None:
            epsilon_values = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        results = {
            'epsilon': [],
            'sensitivity': [],
            'prediction_change': [],
            'confidence_change': []
        }
        
        # Get baseline
        baseline_output = self.model(inputs)
        baseline_pred = baseline_output.argmax(dim=1)
        baseline_conf = F.softmax(baseline_output, dim=1)[0, target].item()
        
        for epsilon in epsilon_values:
            print(f"\nTesting epsilon={epsilon}")
            
            # Define perturbation for this epsilon
            def perturb_fn(inputs: torch.Tensor) -> torch.Tensor:
                noise = torch.FloatTensor(inputs.shape).uniform_(
                    -epsilon, epsilon
                ).to(inputs.device)
                return inputs + noise
            
            # Compute sensitivity at this epsilon
            sens = sensitivity_max(
                explanation_func=attr_method.attribute,
                inputs=inputs,
                perturb_func=perturb_fn,
                perturb_radius=epsilon,
                n_perturb_samples=20
            )
            
            # Test prediction stability
            perturbed = perturb_fn(inputs)
            perturbed_output = self.model(perturbed)
            perturbed_pred = perturbed_output.argmax(dim=1)
            perturbed_conf = F.softmax(perturbed_output, dim=1)[0, target].item()
            
            results['epsilon'].append(epsilon)
            results['sensitivity'].append(sens.mean().item())
            results['prediction_change'].append(
                (baseline_pred != perturbed_pred).float().mean().item()
            )
            results['confidence_change'].append(
                abs(baseline_conf - perturbed_conf)
            )
            
            print(f"  Sensitivity: {sens.mean():.6f}")
            print(f"  Prediction changed: {results['prediction_change'][-1]:.2%}")
            print(f"  Confidence change: {results['confidence_change'][-1]:.4f}")
        
        return results
    
    def identify_security_risks(self,
                               results: Dict,
                               thresholds: Dict = None) -> Dict[str, str]:
        """
        Identify security risks based on metric scores
        
        Returns risk assessment for each attribution method
        """
        if thresholds is None:
            thresholds = {
                'infidelity_high': 0.5,
                'infidelity_critical': 1.0,
                'sensitivity_high': 0.1,
                'sensitivity_critical': 0.5
            }
        
        risk_assessment = {}
        
        for method_name, metrics in results.items():
            risks = []
            risk_level = "LOW"
            
            infid_mean = metrics['infidelity'].mean()
            sens_mean = metrics['sensitivity'].mean()
            
            # Infidelity risks
            if infid_mean > thresholds['infidelity_critical']:
                risks.append("CRITICAL: Very unreliable explanations")
                risk_level = "CRITICAL"
            elif infid_mean > thresholds['infidelity_high']:
                risks.append("HIGH: Unreliable explanations")
                risk_level = "HIGH" if risk_level == "LOW" else risk_level
            
            # Sensitivity risks
            if sens_mean > thresholds['sensitivity_critical']:
                risks.append("CRITICAL: Highly vulnerable to adversarial attacks")
                risk_level = "CRITICAL"
            elif sens_mean > thresholds['sensitivity_high']:
                risks.append("HIGH: Vulnerable to input perturbations")
                risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level
            
            if not risks:
                risks.append("No significant risks detected")
            
            risk_assessment[method_name] = {
                'risk_level': risk_level,
                'risks': risks,
                'infidelity': float(infid_mean),
                'sensitivity': float(sens_mean)
            }
        
        return risk_assessment


# ============================================
# EXAMPLE USAGE
# ============================================

def main():
    # 1. Load pre-trained model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # 2. Prepare image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open('path/to/image.jpg')
    input_tensor = transform(img).unsqueeze(0)
    
    # Get prediction
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    print(f"Predicted class: {predicted_class}")
    
    # 3. Initialize evaluator
    evaluator = ModelRobustnessEvaluator(model)
    
    # 4. Define attribution methods to test
    attribution_methods = {
        'IntegratedGradients': IntegratedGradients(model),
        'Saliency': Saliency(model),
        'InputXGradient': InputXGradient(model),
        'DeepLift': DeepLift(model),
    }
    
    # 5. Evaluate explanation quality
    print("\n" + "="*60)
    print("EVALUATING EXPLANATION QUALITY")
    print("="*60)
    
    results = evaluator.evaluate_explanation_quality(
        inputs=input_tensor,
        attribution_methods=attribution_methods,
        target=predicted_class,
        n_samples=10
    )
    
    # 6. Test adversarial robustness
    print("\n" + "="*60)
    print("TESTING ADVERSARIAL ROBUSTNESS")
    print("="*60)
    
    adv_results = evaluator.test_adversarial_robustness(
        inputs=input_tensor,
        attr_method=attribution_methods['IntegratedGradients'],
        target=predicted_class,
        epsilon_values=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    )
    
    # 7. Identify security risks
    print("\n" + "="*60)
    print("SECURITY RISK ASSESSMENT")
    print("="*60)
    
    risk_assessment = evaluator.identify_security_risks(results)
    
    for method, assessment in risk_assessment.items():
        print(f"\n{method}:")
        print(f"  Risk Level: {assessment['risk_level']}")
        print(f"  Infidelity: {assessment['infidelity']:.6f}")
        print(f"  Sensitivity: {assessment['sensitivity']:.6f}")
        print(f"  Risks:")
        for risk in assessment['risks']:
            print(f"    - {risk}")
    
    return results, adv_results, risk_assessment


if __name__ == "__main__":
    results, adv_results, risk_assessment = main()