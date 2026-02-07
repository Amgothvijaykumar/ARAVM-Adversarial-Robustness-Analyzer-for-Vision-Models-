"""
================================================================================
ADVERSARIAL ROBUSTNESS ANALYZER FOR VISION MODELS (ARAVM)
================================================================================
A unified security evaluation framework for PyTorch vision models.

Integration Flow:
    Level 1 (White-Box): ART FGSM/PGD + Captum gradient heatmaps
    Level 2 (Black-Box): ART HopSkipJump / Query-based attacks
    Level 3 (Patch/Occlusion): phattacks ROA (Rectangular Occlusion Attack)
    Level 4 (Mitigation): ART JPEG compression + defensive measures

Author: ARAVM Framework
Version: 1.0.0
================================================================================
"""

# =============================================================================
# SSL CERTIFICATE FIX (for macOS Python installations)
# =============================================================================
import ssl
import certifi
import os

# Fix SSL certificate verification for macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import warnings

# =============================================================================
# ART IMPORTS (Adversarial Robustness Toolbox)
# =============================================================================
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,      # FGSM - Level 1
    ProjectedGradientDescent, # PGD - Level 1
    HopSkipJump,             # Black-box - Level 2
)
from art.defences.preprocessor import JpegCompression, SpatialSmoothing

# =============================================================================
# CAPTUM IMPORTS (Interpretability & Heatmaps)
# =============================================================================
from captum.attr import (
    IntegratedGradients,
    Saliency,
    GradientShap,
    Occlusion as CaptumOcclusion,
)
from captum.attr import visualization as viz

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================
@dataclass
class AttackConfig:
    """Configuration for adversarial attacks"""
    # FGSM/PGD parameters
    epsilon: float = 0.03          # Perturbation budget (0.01 - 0.3)
    eps_step: float = 0.01         # Step size for PGD
    max_iter: int = 40             # Max iterations for PGD
    targeted: bool = False         # Targeted vs untargeted attack
    
    # Patch attack parameters (ROA)
    patch_width: int = 100
    patch_height: int = 50
    patch_x_skip: int = 4          # Stride for gradient search
    patch_y_skip: int = 4
    potential_positions: int = 10  # Top-k positions to evaluate
    
    # Black-box parameters
    max_queries: int = 1000
    
@dataclass
class DefenseConfig:
    """Configuration for defensive measures"""
    jpeg_quality: int = 50         # JPEG compression quality (1-95)
    spatial_window: int = 3        # Spatial smoothing window size
    
@dataclass
class MetricReport:
    """Container for robustness metrics"""
    # Model Capability Metrics
    clean_accuracy: float = 0.0
    clean_confidence: float = 0.0
    clean_f1: float = 0.0
    
    # Attack Effectiveness Metrics
    misclassification_ratio: float = 0.0   # MR
    targeted_attack_success: float = 0.0   # TAS
    avg_confidence_change: float = 0.0     # ACC
    
    # Attack Cost Metrics
    avg_l2_distortion: float = 0.0
    avg_linf_distortion: float = 0.0
    
    # Defense Effectiveness
    accuracy_after_defense: float = 0.0
    accuracy_variance: float = 0.0


# =============================================================================
# VICTIM MODEL WRAPPER
# =============================================================================
class VictimModel:
    """
    Wrapper for the target vision model.
    Supports easy swapping of model architectures.
    """
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
        device: str = None
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_name, pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        # Create ART classifier wrapper
        self.art_classifier = self._create_art_classifier()
        
        # Standard transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        self.normalize = transforms.Normalize(
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_STD
        )
        
        print(f"[✓] Victim Model Loaded: {model_name}")
        print(f"    Device: {self.device}")
        print(f"    Classes: {num_classes}")
        
    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """Load a torchvision model by name"""
        model_loaders = {
            "resnet50": models.resnet50,
            "resnet18": models.resnet18,
            "resnet101": models.resnet101,
            "vgg16": models.vgg16,
            "densenet121": models.densenet121,
            "mobilenet_v2": models.mobilenet_v2,
            "efficientnet_b0": models.efficientnet_b0,
        }
        
        if model_name not in model_loaders:
            raise ValueError(f"Unsupported model: {model_name}. Options: {list(model_loaders.keys())}")
        
        # Load with weights parameter (modern PyTorch)
        try:
            weights = "DEFAULT" if pretrained else None
            model = model_loaders[model_name](weights=weights)
        except TypeError:
            # Fallback for older PyTorch versions
            model = model_loaders[model_name](pretrained=pretrained)
            
        return model
    
    def _create_art_classifier(self) -> PyTorchClassifier:
        """Wrap the PyTorch model with ART's classifier"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        classifier = PyTorchClassifier(
            model=self.model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, 224, 224),
            nb_classes=self.num_classes,
            clip_values=(0.0, 1.0),
            channels_first=True,
            preprocessing=(self.IMAGENET_MEAN, self.IMAGENET_STD),
        )
        return classifier
    
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> Tuple[int, float]:
        """Get prediction and confidence for input"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        
        with torch.no_grad():
            # Apply normalization
            x_norm = self.normalize(x)
            output = self.model(x_norm)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            
        return pred_class, confidence
    
    def predict_batch(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction with ART classifier"""
        predictions = self.art_classifier.predict(x)
        pred_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        return pred_classes, confidences


# =============================================================================
# LEVEL 1: WHITE-BOX GRADIENT ATTACKS (ART + Captum)
# =============================================================================
class WhiteBoxAttacker:
    """
    Level 1 Security Analysis: White-Box Gradient Attacks
    Uses ART for FGSM/PGD and Captum for gradient visualization
    """
    
    def __init__(self, victim: VictimModel, config: AttackConfig = None):
        self.victim = victim
        self.config = config or AttackConfig()
        
        # Initialize attack algorithms
        self.fgsm = FastGradientMethod(
            estimator=victim.art_classifier,
            eps=self.config.epsilon,
            targeted=self.config.targeted
        )
        
        self.pgd = ProjectedGradientDescent(
            estimator=victim.art_classifier,
            eps=self.config.epsilon,
            eps_step=self.config.eps_step,
            max_iter=self.config.max_iter,
            targeted=self.config.targeted
        )
        
        # Captum attribution methods
        self.integrated_gradients = IntegratedGradients(victim.model)
        self.saliency = Saliency(victim.model)
        
        print("[✓] Level 1 (White-Box) Attacker Initialized")
        
    def fgsm_attack(self, x: np.ndarray, epsilon: float = None) -> np.ndarray:
        """
        Fast Gradient Sign Method attack
        The "Noise Slider" - adjustable epsilon from 0.01 to 0.3
        """
        if epsilon is not None:
            # Temporarily update epsilon
            attack = FastGradientMethod(
                estimator=self.victim.art_classifier,
                eps=epsilon,
                targeted=self.config.targeted
            )
            return attack.generate(x=x)
        return self.fgsm.generate(x=x)
    
    def pgd_attack(self, x: np.ndarray) -> np.ndarray:
        """Projected Gradient Descent - stronger iterative attack"""
        return self.pgd.generate(x=x)
    
    def generate_heatmap(
        self,
        x: torch.Tensor,
        target_class: int = None,
        method: str = "integrated_gradients"
    ) -> np.ndarray:
        """
        Generate attribution heatmap using Captum
        Returns normalized attribution map
        """
        # Ensure correct shape: [1, 3, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Ensure we only have 3 channels (take first 3 if more)
        if x.shape[1] != 3:
            x = x[:, :3, :, :]
        
        x = x.clone().detach().to(self.victim.device)
        x.requires_grad = True
        
        if target_class is None:
            # Use predicted class as target
            with torch.no_grad():
                target_class, _ = self.victim.predict(x.squeeze(0))
        
        # Normalize input for model
        x_norm = self.victim.normalize(x)
        
        if method == "integrated_gradients":
            attributions = self.integrated_gradients.attribute(
                x_norm, target=target_class, n_steps=50
            )
        elif method == "saliency":
            attributions = self.saliency.attribute(x_norm, target=target_class)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to numpy and normalize
        attr = attributions.squeeze().cpu().detach().numpy()
        attr = np.transpose(attr, (1, 2, 0))  # CHW -> HWC
        
        return attr
    
    def compare_heatmaps(
        self,
        x_clean: np.ndarray,
        x_adv: np.ndarray,
        save_path: str = None
    ) -> plt.Figure:
        """
        Generate side-by-side comparison of Clean vs Attacked heatmaps
        Visualizes how the attack shifts model attention
        """
        # Convert to tensors
        x_clean_t = torch.from_numpy(x_clean).float()
        x_adv_t = torch.from_numpy(x_adv).float()
        
        if x_clean_t.dim() == 4:
            x_clean_t = x_clean_t.squeeze(0)
            x_adv_t = x_adv_t.squeeze(0)
        
        # Get predictions
        clean_pred, clean_conf = self.victim.predict(x_clean_t)
        adv_pred, adv_conf = self.victim.predict(x_adv_t)
        
        # Generate heatmaps
        clean_attr = self.generate_heatmap(x_clean_t, clean_pred)
        adv_attr = self.generate_heatmap(x_adv_t, adv_pred)
        
        # Prepare images for visualization
        clean_img = np.transpose(x_clean_t.numpy(), (1, 2, 0))
        adv_img = np.transpose(x_adv_t.numpy(), (1, 2, 0))
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Level 1: White-Box Attack Analysis", fontsize=14, fontweight='bold')
        
        # Row 1: Clean
        axes[0, 0].imshow(np.clip(clean_img, 0, 1))
        axes[0, 0].set_title(f"Clean Image\nPred: {clean_pred} ({clean_conf:.2%})")
        axes[0, 0].axis('off')
        
        # Normalize attribution for visualization
        clean_attr_norm = np.sum(np.abs(clean_attr), axis=2)
        axes[0, 1].imshow(clean_attr_norm, cmap='hot')
        axes[0, 1].set_title("Clean Heatmap")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.clip(clean_img, 0, 1))
        axes[0, 2].imshow(clean_attr_norm, cmap='hot', alpha=0.5)
        axes[0, 2].set_title("Clean Overlay")
        axes[0, 2].axis('off')
        
        # Row 2: Adversarial
        axes[1, 0].imshow(np.clip(adv_img, 0, 1))
        axes[1, 0].set_title(f"Adversarial Image\nPred: {adv_pred} ({adv_conf:.2%})")
        axes[1, 0].axis('off')
        
        adv_attr_norm = np.sum(np.abs(adv_attr), axis=2)
        axes[1, 1].imshow(adv_attr_norm, cmap='hot')
        axes[1, 1].set_title("Attacked Heatmap")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(np.clip(adv_img, 0, 1))
        axes[1, 2].imshow(adv_attr_norm, cmap='hot', alpha=0.5)
        axes[1, 2].set_title("Attacked Overlay")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[✓] Heatmap comparison saved: {save_path}")
            
        return fig


# =============================================================================
# LEVEL 2: BLACK-BOX ATTACKS
# =============================================================================
class BlackBoxAttacker:
    """
    Level 2 Security Analysis: Black-Box Query-Based Attacks
    Uses ART's HopSkipJump for decision-based attacks
    """
    
    def __init__(self, victim: VictimModel, config: AttackConfig = None):
        self.victim = victim
        self.config = config or AttackConfig()
        
        # HopSkipJump - Decision-based black-box attack
        self.hop_skip_jump = HopSkipJump(
            classifier=victim.art_classifier,
            targeted=self.config.targeted,
            max_iter=50,
            max_eval=self.config.max_queries,
            init_eval=100,
        )
        
        print("[✓] Level 2 (Black-Box) Attacker Initialized")
        
    def hop_skip_jump_attack(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        HopSkipJump attack - only requires query access
        Returns adversarial example and number of queries used
        """
        x_adv = self.hop_skip_jump.generate(x=x)
        # Note: Actual query count would need to be tracked in ART
        return x_adv, self.config.max_queries


# =============================================================================
# LEVEL 3: PATCH / OCCLUSION ATTACKS (phattacks ROA)
# =============================================================================
class PatchAttacker:
    """
    Level 3 Security Analysis: Localized Patch Attacks
    Implements ROA (Rectangular Occlusion Attack) from phattacks
    """
    
    def __init__(self, victim: VictimModel, config: AttackConfig = None):
        self.victim = victim
        self.config = config or AttackConfig()
        self.device = victim.device
        
        print("[✓] Level 3 (Patch/ROA) Attacker Initialized")
        
    def gradient_based_search(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        width: int = None,
        height: int = None
    ) -> Tuple[int, int]:
        """
        ROA: Find the highest-gradient rectangle area for patch placement
        This is the "Smart Patch" logic from /phattacks
        
        Returns:
            (best_x, best_y): Top-left coordinates of optimal patch position
        """
        width = width or self.config.patch_width
        height = height or self.config.patch_height
        x_skip = self.config.patch_x_skip
        y_skip = self.config.patch_y_skip
        potential_nums = self.config.potential_positions
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        _, _, img_h, img_w = x.shape
        
        # Compute gradients with respect to input
        x_grad = x.clone().detach().requires_grad_(True)
        x_norm = self.victim.normalize(x_grad)
        
        output = self.victim.model(x_norm)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        gradient = x_grad.grad.detach()
        
        # Normalize gradients
        max_val = torch.abs(gradient).view(gradient.shape[0], -1).max(dim=1)[0]
        gradient = gradient / (max_val[:, None, None, None] + 1e-8)
        
        # Calculate number of positions to search
        x_times = (img_w - width) // x_skip
        y_times = (img_h - height) // y_skip
        
        # Calculate gradient magnitude for each potential position
        position_scores = []
        
        for i in range(x_times):
            for j in range(y_times):
                patch_gradient = gradient[:, :, 
                                         y_skip * j:y_skip * j + height,
                                         x_skip * i:x_skip * i + width]
                # Sum of squared gradients (L2 norm)
                score = torch.sum(patch_gradient ** 2).item()
                position_scores.append((score, x_skip * i, y_skip * j))
        
        # Sort by score and get top-k positions
        position_scores.sort(reverse=True, key=lambda x: x[0])
        top_positions = position_scores[:potential_nums]
        
        # Validate positions by testing actual occlusion
        best_loss = -float('inf')
        best_pos = (0, 0)
        
        with torch.no_grad():
            for score, px, py in top_positions:
                # Create occluded version
                x_occluded = x.clone()
                x_occluded[:, :, py:py+height, px:px+width] = 0.5  # Gray patch
                
                x_norm = self.victim.normalize(x_occluded)
                output = self.victim.model(x_norm)
                loss = nn.CrossEntropyLoss()(output, y)
                
                if loss.item() > best_loss:
                    best_loss = loss.item()
                    best_pos = (px, py)
        
        return best_pos
    
    def apply_adversarial_patch(
        self,
        x: np.ndarray,
        position: Tuple[int, int] = None,
        patch_value: float = 0.5,
        optimize: bool = True
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Apply adversarial patch at optimal or specified position
        
        Args:
            x: Input image (NCHW or CHW format)
            position: (x, y) position or None for auto-search
            patch_value: Patch color value (0-1)
            optimize: Whether to optimize patch content
            
        Returns:
            (adversarial_image, patch_position)
        """
        x_t = torch.from_numpy(x).float()
        
        if x_t.dim() == 3:
            x_t = x_t.unsqueeze(0)
            
        # Get true label
        pred_class, _ = self.victim.predict(x_t.squeeze(0))
        y = torch.tensor([pred_class]).to(self.device)
        
        # Find optimal position if not specified
        if position is None:
            position = self.gradient_based_search(x_t, y)
            print(f"    [ROA] Optimal patch position: {position}")
        
        px, py = position
        width = self.config.patch_width
        height = self.config.patch_height
        
        # Apply patch
        x_patched = x_t.clone()
        
        if optimize:
            # Optimize patch content using PGD
            x_patched = self._optimize_patch(x_patched, y, px, py, width, height)
        else:
            # Simple gray patch
            x_patched[:, :, py:py+height, px:px+width] = patch_value
        
        return x_patched.squeeze(0).numpy(), position
    
    def _optimize_patch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px: int, py: int,
        width: int, height: int,
        num_iter: int = 20,
        alpha: float = 0.01
    ) -> torch.Tensor:
        """Optimize patch content using PGD in the patch region"""
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Initialize patch randomly
        patch = torch.rand(1, 3, height, width).to(self.device)
        
        for _ in range(num_iter):
            patch.requires_grad = True
            
            # Apply patch
            x_patched = x.clone()
            x_patched[:, :, py:py+height, px:px+width] = patch
            
            # Forward pass
            x_norm = self.victim.normalize(x_patched)
            output = self.victim.model(x_norm)
            
            # Maximize loss (cause misclassification)
            loss = -nn.CrossEntropyLoss()(output, y)
            loss.backward()
            
            # Update patch
            with torch.no_grad():
                patch = patch - alpha * patch.grad.sign()
                patch = torch.clamp(patch, 0, 1)
                
        # Apply optimized patch
        x_result = x.clone()
        x_result[:, :, py:py+height, px:px+width] = patch.detach()
        
        return x_result


# =============================================================================
# LEVEL 4: DEFENSIVE MITIGATIONS
# =============================================================================
class DefensiveAnalyzer:
    """
    Level 4: Defensive Measures and Hardening
    Uses ART's preprocessing defenses
    """
    
    def __init__(self, victim: VictimModel, config: DefenseConfig = None):
        self.victim = victim
        self.config = config or DefenseConfig()
        
        # JPEG Compression Defense
        self.jpeg_defense = JpegCompression(
            clip_values=(0.0, 1.0),
            quality=self.config.jpeg_quality,
            channels_first=True,
            apply_fit=False,
            apply_predict=True,
        )
        
        # Spatial Smoothing Defense
        self.spatial_defense = SpatialSmoothing(
            window_size=self.config.spatial_window,
            channels_first=True,
            clip_values=(0.0, 1.0),
        )
        
        print("[✓] Level 4 (Defense) Analyzer Initialized")
        
    def apply_jpeg_defense(self, x: np.ndarray) -> np.ndarray:
        """Apply JPEG compression to remove adversarial noise"""
        # JPEG requires values in [0, 1] range
        x_clipped = np.clip(x, 0.0, 1.0)
        x_defended, _ = self.jpeg_defense(x_clipped)
        return x_defended
    
    def apply_spatial_smoothing(self, x: np.ndarray) -> np.ndarray:
        """Apply spatial smoothing to blur adversarial perturbations"""
        # Ensure values are in valid range
        x_clipped = np.clip(x, 0.0, 1.0)
        x_defended, _ = self.spatial_defense(x_clipped)
        return x_defended
    
    def evaluate_defense(
        self,
        x_clean: np.ndarray,
        x_adv: np.ndarray,
        y_true: int
    ) -> Dict[str, float]:
        """
        Evaluate defense effectiveness
        
        Returns metrics comparing accuracy with and without defense
        """
        results = {}
        
        # Ensure input is in correct format
        if x_adv.ndim == 4:
            x_adv_single = x_adv[0]
        else:
            x_adv_single = x_adv
        
        # Without defense
        pred_adv, conf_adv = self.victim.predict(
            torch.from_numpy(x_adv_single).float()
        )
        results['no_defense'] = {
            'correct': pred_adv == y_true,
            'confidence': conf_adv
        }
        
        # With JPEG defense
        try:
            x_jpeg = self.apply_jpeg_defense(x_adv)
            if x_jpeg.ndim == 4:
                x_jpeg = x_jpeg[0]
            pred_jpeg, conf_jpeg = self.victim.predict(
                torch.from_numpy(x_jpeg).float()
            )
            results['jpeg'] = {
                'correct': pred_jpeg == y_true,
                'confidence': conf_jpeg
            }
        except Exception as e:
            results['jpeg'] = {
                'correct': False,
                'confidence': 0.0,
                'error': str(e)
            }
        
        # With spatial smoothing
        try:
            x_smooth = self.apply_spatial_smoothing(x_adv)
            if x_smooth.ndim == 4:
                x_smooth = x_smooth[0]
            pred_smooth, conf_smooth = self.victim.predict(
                torch.from_numpy(x_smooth).float()
            )
            results['spatial_smoothing'] = {
                'correct': pred_smooth == y_true,
                'confidence': conf_smooth
            }
        except Exception as e:
            results['spatial_smoothing'] = {
                'correct': False,
                'confidence': 0.0,
                'error': str(e)
            }
        
        return results


# =============================================================================
# ROBUSTNESS METRICS CALCULATOR
# =============================================================================
class MetricsCalculator:
    """
    Calculate comprehensive robustness metrics as defined in the
    Security Evaluation Framework for Inference (SEFI)
    """
    
    def __init__(self, victim: VictimModel):
        self.victim = victim
        
    def calculate_metrics(
        self,
        x_clean: np.ndarray,
        x_adv: np.ndarray,
        y_true: np.ndarray
    ) -> MetricReport:
        """
        Calculate full metric report for a batch of images
        
        Args:
            x_clean: Clean images (N, C, H, W)
            x_adv: Adversarial images (N, C, H, W)
            y_true: True labels (N,)
        """
        report = MetricReport()
        
        # Ensure batch dimension
        if x_clean.ndim == 3:
            x_clean = x_clean[np.newaxis, ...]
            x_adv = x_adv[np.newaxis, ...]
            y_true = np.array([y_true])
        
        n_samples = len(y_true)
        
        # Model Capability Metrics (Clean)
        clean_preds, clean_confs = self.victim.predict_batch(x_clean)
        report.clean_accuracy = np.mean(clean_preds == y_true)
        report.clean_confidence = np.mean(clean_confs)
        
        # Attack Effectiveness Metrics
        adv_preds, adv_confs = self.victim.predict_batch(x_adv)
        
        # Misclassification Ratio (MR)
        report.misclassification_ratio = np.mean(adv_preds != y_true)
        
        # Average Confidence Change (ACC)
        report.avg_confidence_change = np.mean(np.abs(clean_confs - adv_confs))
        
        # Attack Cost Metrics (Distortion)
        perturbation = x_adv - x_clean
        
        # L2 distortion (per sample)
        l2_norms = np.sqrt(np.sum(perturbation ** 2, axis=(1, 2, 3)))
        report.avg_l2_distortion = np.mean(l2_norms)
        
        # L∞ distortion
        linf_norms = np.max(np.abs(perturbation), axis=(1, 2, 3))
        report.avg_linf_distortion = np.mean(linf_norms)
        
        return report
    
    def print_report(self, report: MetricReport, attack_name: str = "Attack"):
        """Pretty print the metric report"""
        print("\n" + "=" * 60)
        print(f"  ROBUSTNESS METRICS REPORT: {attack_name}")
        print("=" * 60)
        
        print("\n📊 MODEL CAPABILITY METRICS (Baseline)")
        print("-" * 40)
        print(f"  Clean Accuracy:     {report.clean_accuracy:.2%}")
        print(f"  Clean Confidence:   {report.clean_confidence:.2%}")
        
        print("\n⚔️  ATTACK EFFECTIVENESS METRICS")
        print("-" * 40)
        print(f"  Misclassification Ratio (MR): {report.misclassification_ratio:.2%}")
        print(f"  Avg Confidence Change (ACC):  {report.avg_confidence_change:.4f}")
        
        print("\n📏 PERTURBATION METRICS")
        print("-" * 40)
        print(f"  Average L2 Distortion:   {report.avg_l2_distortion:.4f}")
        print(f"  Average L∞ Distortion:   {report.avg_linf_distortion:.4f}")
        
        print("\n" + "=" * 60)


# =============================================================================
# MAIN ANALYZER - UNIFIED EXECUTION FLOW
# =============================================================================
class AdversarialRobustnessAnalyzer:
    """
    ARAVM: Main orchestrator for the 4-Level security audit
    Integrates ART, Captum, and phattacks into a single execution flow
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        attack_config: AttackConfig = None,
        defense_config: DefenseConfig = None,
        device: str = None
    ):
        print("\n" + "=" * 60)
        print("  ADVERSARIAL ROBUSTNESS ANALYZER FOR VISION MODELS (ARAVM)")
        print("=" * 60 + "\n")
        
        # Initialize victim model
        self.victim = VictimModel(
            model_name=model_name,
            device=device
        )
        
        # Initialize attack levels
        self.attack_config = attack_config or AttackConfig()
        self.defense_config = defense_config or DefenseConfig()
        
        self.level1 = WhiteBoxAttacker(self.victim, self.attack_config)
        self.level2 = BlackBoxAttacker(self.victim, self.attack_config)
        self.level3 = PatchAttacker(self.victim, self.attack_config)
        self.level4 = DefensiveAnalyzer(self.victim, self.defense_config)
        
        # Metrics calculator
        self.metrics = MetricsCalculator(self.victim)
        
        print("\n[✓] ARAVM Initialization Complete\n")
        
    def load_sample_image(self, image_path: str = None) -> Tuple[np.ndarray, int]:
        """
        Load a sample image for testing
        If no path provided, generates a random test image
        """
        if image_path:
            img = Image.open(image_path).convert('RGB')
            x = self.victim.transform(img).numpy()
        else:
            # Generate random test image
            print("[!] No image path provided, using random noise image")
            x = np.random.rand(3, 224, 224).astype(np.float32)
        
        # Get true prediction as ground truth
        pred, conf = self.victim.predict(torch.from_numpy(x))
        
        return x, pred
    
    def noise_slider(
        self,
        x: np.ndarray,
        epsilon_values: List[float] = None
    ) -> Dict[float, Dict]:
        """
        The "Noise Slider" - Run FGSM with adjustable epsilon
        
        Args:
            x: Input image
            epsilon_values: List of epsilon values to test (0.01 to 0.3)
            
        Returns:
            Dictionary mapping epsilon to attack results
        """
        if epsilon_values is None:
            epsilon_values = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
        
        results = {}
        x_batch = x[np.newaxis, ...] if x.ndim == 3 else x
        
        # Get clean prediction
        clean_pred, clean_conf = self.victim.predict(torch.from_numpy(x))
        
        print("\n" + "=" * 60)
        print("  NOISE SLIDER: FGSM Attack Analysis")
        print("=" * 60)
        print(f"  Clean Prediction: Class {clean_pred} ({clean_conf:.2%})")
        print("-" * 60)
        
        for eps in epsilon_values:
            x_adv = self.level1.fgsm_attack(x_batch, epsilon=eps)
            adv_pred, adv_conf = self.victim.predict(
                torch.from_numpy(x_adv.squeeze(0))
            )
            
            success = adv_pred != clean_pred
            l2_dist = np.sqrt(np.sum((x_adv - x_batch) ** 2))
            
            results[eps] = {
                'adversarial': x_adv,
                'prediction': adv_pred,
                'confidence': adv_conf,
                'success': success,
                'l2_distortion': l2_dist
            }
            
            status = "✗ FOOLED" if success else "✓ Robust"
            print(f"  ε={eps:.2f}: Pred={adv_pred} ({adv_conf:.2%}) | L2={l2_dist:.3f} | {status}")
        
        print("-" * 60)
        return results
    
    def run_full_audit(
        self,
        x: np.ndarray,
        y_true: int,
        save_visualizations: bool = True,
        output_dir: str = "."
    ) -> Dict[str, MetricReport]:
        """
        Run complete 4-Level security audit on a single image
        
        Args:
            x: Input image (CHW format)
            y_true: True label
            save_visualizations: Whether to save visualization plots
            output_dir: Directory for saving outputs
            
        Returns:
            Dictionary of MetricReports for each attack level
        """
        reports = {}
        x_batch = x[np.newaxis, ...].astype(np.float32)
        y_batch = np.array([y_true])
        
        print("\n" + "=" * 70)
        print("  FULL SECURITY AUDIT - 4 LEVEL ANALYSIS")
        print("=" * 70)
        
        # =================================================================
        # LEVEL 1: White-Box Attacks
        # =================================================================
        print("\n[LEVEL 1] WHITE-BOX GRADIENT ATTACKS")
        print("-" * 50)
        
        # FGSM Attack
        x_fgsm = self.level1.fgsm_attack(x_batch)
        report_fgsm = self.metrics.calculate_metrics(x_batch, x_fgsm, y_batch)
        reports['fgsm'] = report_fgsm
        self.metrics.print_report(report_fgsm, "FGSM (ε=0.03)")
        
        # PGD Attack
        x_pgd = self.level1.pgd_attack(x_batch)
        report_pgd = self.metrics.calculate_metrics(x_batch, x_pgd, y_batch)
        reports['pgd'] = report_pgd
        self.metrics.print_report(report_pgd, "PGD (40 iterations)")
        
        # Generate heatmap comparison
        if save_visualizations:
            fig = self.level1.compare_heatmaps(
                x_batch, x_pgd,
                save_path=f"{output_dir}/level1_heatmap_comparison.png"
            )
            plt.close(fig)
        
        # =================================================================
        # LEVEL 2: Black-Box Attack (Skip for speed in demo)
        # =================================================================
        print("\n[LEVEL 2] BLACK-BOX QUERY ATTACKS")
        print("-" * 50)
        print("  [!] Skipping HopSkipJump (time-intensive)")
        print("      Enable with: analyzer.level2.hop_skip_jump_attack(x)")
        
        # =================================================================
        # LEVEL 3: Patch Attack (ROA)
        # =================================================================
        print("\n[LEVEL 3] PATCH/OCCLUSION ATTACKS (ROA)")
        print("-" * 50)
        
        x_patched, patch_pos = self.level3.apply_adversarial_patch(x_batch)
        x_patched_batch = x_patched[np.newaxis, ...]
        report_patch = self.metrics.calculate_metrics(x_batch, x_patched_batch, y_batch)
        reports['patch'] = report_patch
        self.metrics.print_report(report_patch, f"ROA Patch @ {patch_pos}")
        
        # =================================================================
        # LEVEL 4: Defense Evaluation
        # =================================================================
        print("\n[LEVEL 4] DEFENSE EVALUATION")
        print("-" * 50)
        
        defense_results = self.level4.evaluate_defense(
            x_batch, x_pgd, y_true
        )
        
        print("\n  Defense Effectiveness against PGD:")
        for defense_name, result in defense_results.items():
            status = "✓ RECOVERED" if result['correct'] else "✗ Still Fooled"
            print(f"    {defense_name:20}: {status} (conf={result['confidence']:.2%})")
        
        # =================================================================
        # VISUALIZATION DASHBOARD
        # =================================================================
        if save_visualizations:
            self._create_dashboard(
                x_batch[0], x_fgsm[0], x_patched,
                save_path=f"{output_dir}/aravm_dashboard.png"
            )
        
        print("\n" + "=" * 70)
        print("  AUDIT COMPLETE")
        print("=" * 70 + "\n")
        
        return reports
    
    def _create_dashboard(
        self,
        x_clean: np.ndarray,
        x_noise: np.ndarray,
        x_patch: np.ndarray,
        save_path: str = None
    ):
        """
        Create visualization dashboard:
        [Original] -> [Noise Attack] -> [Patch Attack] -> [Heatmap]
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle("ARAVM Security Analysis Dashboard", fontsize=16, fontweight='bold')
        
        # Helper to convert CHW to HWC
        def to_image(x):
            return np.clip(np.transpose(x, (1, 2, 0)), 0, 1)
        
        # Original
        clean_pred, clean_conf = self.victim.predict(torch.from_numpy(x_clean))
        axes[0].imshow(to_image(x_clean))
        axes[0].set_title(f"Original\nClass: {clean_pred} ({clean_conf:.1%})")
        axes[0].axis('off')
        
        # Noise Attack (FGSM)
        noise_pred, noise_conf = self.victim.predict(torch.from_numpy(x_noise))
        axes[1].imshow(to_image(x_noise))
        axes[1].set_title(f"FGSM Attack\nClass: {noise_pred} ({noise_conf:.1%})")
        axes[1].axis('off')
        
        # Patch Attack
        patch_pred, patch_conf = self.victim.predict(torch.from_numpy(x_patch))
        axes[2].imshow(to_image(x_patch))
        axes[2].set_title(f"Patch Attack\nClass: {patch_pred} ({patch_conf:.1%})")
        axes[2].axis('off')
        
        # Perturbation Visualization
        perturbation = np.abs(x_noise - x_clean)
        perturbation = perturbation / (perturbation.max() + 1e-8)  # Normalize
        axes[3].imshow(to_image(perturbation))
        axes[3].set_title("Perturbation Magnified")
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[✓] Dashboard saved: {save_path}")
        
        return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """
    Example usage of ARAVM
    Run a full security audit on a sample image
    """
    # Initialize analyzer with ResNet-50 victim
    analyzer = AdversarialRobustnessAnalyzer(
        model_name="resnet50",
        attack_config=AttackConfig(
            epsilon=0.03,
            patch_width=100,
            patch_height=50
        ),
        defense_config=DefenseConfig(
            jpeg_quality=50
        )
    )
    
    # Load or generate sample image
    # For demo, using random noise (replace with real image path)
    x, y_true = analyzer.load_sample_image()
    
    print(f"\n[*] Sample loaded: Shape={x.shape}, True Label={y_true}")
    
    # Run noise slider analysis
    slider_results = analyzer.noise_slider(x)
    
    # Run full 4-level audit
    reports = analyzer.run_full_audit(
        x, y_true,
        save_visualizations=True,
        output_dir="/Users/amgothvijaykumar/Projects/GAN_toolBox"
    )
    
    return analyzer, reports


if __name__ == "__main__":
    analyzer, reports = main()
