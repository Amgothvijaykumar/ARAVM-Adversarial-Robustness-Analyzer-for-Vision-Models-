import torch
import numpy as np
from captum.metrics import infidelity_perturb_func_decorator

# ============================================
# PERTURBATION FUNCTION 1: Gaussian Noise
# ============================================

def gaussian_perturb_fn(inputs: torch.Tensor,
                       baselines: torch.Tensor = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise to inputs
    Returns: (perturbations, perturbed_inputs)
    """
    noise_std = 0.1
    noise = torch.randn_like(inputs) * noise_std
    perturbed_inputs = inputs + noise
    
    # Compute perturbation magnitude
    if baselines is not None:
        perturbations = (inputs - perturbed_inputs) / (inputs - baselines + 1e-8)
    else:
        perturbations = inputs - perturbed_inputs
    
    return perturbations, perturbed_inputs


# ============================================
# PERTURBATION FUNCTION 2: Using Decorator
# ============================================

@infidelity_perturb_func_decorator(multiply_by_inputs=True)
def simple_noise_perturb(inputs: torch.Tensor) -> torch.Tensor:
    """
    Decorator automatically handles perturbation computation
    Just return perturbed inputs
    """
    noise = torch.randn_like(inputs) * 0.05
    return inputs + noise


# ============================================
# PERTURBATION FUNCTION 3: Patch-based
# ============================================

def patch_perturb_fn(inputs: torch.Tensor,
                    patch_size: int = 16
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly occlude patches (useful for images)
    """
    perturbed = inputs.clone()
    batch_size, channels, height, width = inputs.shape
    
    # Random patch location
    y = np.random.randint(0, height - patch_size + 1)
    x = np.random.randint(0, width - patch_size + 1)
    
    # Occlude patch
    perturbed[:, :, y:y+patch_size, x:x+patch_size] = 0
    
    perturbations = inputs - perturbed
    
    return perturbations, perturbed


# ============================================
# PERTURBATION FUNCTION 4: Adversarial (FGSM)
# ============================================

def fgsm_perturb_fn(model: torch.nn.Module,
                   epsilon: float = 0.03):
    """
    Fast Gradient Sign Method perturbations
    Tests robustness against adversarial attacks
    """
    def perturb(inputs: torch.Tensor,
               target: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_grad = inputs.clone().requires_grad_(True)
        
        output = model(inputs_grad)
        
        if target is None:
            target = output.argmax(dim=1)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # FGSM perturbation
        perturbation = epsilon * inputs_grad.grad.sign()
        perturbed_inputs = inputs + perturbation
        
        return perturbation, perturbed_inputs
    
    return perturb


# ============================================
# PERTURBATION FUNCTION 5: Blur/Smooth
# ============================================

def blur_perturb_fn(inputs: torch.Tensor,
                   kernel_size: int = 5
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Gaussian blur (tests robustness to smoothing)
    """
    from torchvision.transforms import functional as TF
    
    perturbed = TF.gaussian_blur(inputs, kernel_size=[kernel_size, kernel_size])
    perturbations = inputs - perturbed
    
    return perturbations, perturbed


# ============================================
# USAGE EXAMPLES
# ============================================

# Example 1: Using with infidelity
from captum.metrics import infidelity
from captum.attr import IntegratedGradients

model = models.resnet50(pretrained=True)
ig = IntegratedGradients(model)

inputs = torch.randn(2, 3, 224, 224)
attributions = ig.attribute(inputs, target=0)

# Compute infidelity with different perturbations
infid_gaussian = infidelity(
    model,
    gaussian_perturb_fn,
    inputs,
    attributions,
    target=0,
    n_perturb_samples=10
)

infid_simple = infidelity(
    model,
    simple_noise_perturb,
    inputs,
    attributions,
    target=0,
    n_perturb_samples=10
)

print(f"Infidelity (Gaussian): {infid_gaussian.mean():.4f}")
print(f"Infidelity (Simple): {infid_simple.mean():.4f}")