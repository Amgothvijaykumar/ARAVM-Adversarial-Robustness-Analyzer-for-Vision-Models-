"""
Complete example: Train and test an adversarial patch
"""

import torch
import torchvision
import torchvision.transforms as transforms
from art.attacks.evasion.adversarial_patch import AdversarialPatchPyTorch
from art.estimators.classification import PyTorchClassifier
import numpy as np

def create_adversarial_patch_complete_example():
    """
    Complete workflow for creating and testing an adversarial patch.
    """
    
    # 1. Setup classifier
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    
    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
    )
    
    # 2. Load training data (use real ImageNet or CIFAR samples)
    # For this example, using random data
    num_train = 50
    x_train = np.random.rand(num_train, 3, 224, 224).astype(np.float32)
    
    # 3. Create target labels (untargeted attack - fool to any wrong class)
    # For targeted attack, set specific target class
    y_train_orig = classifier.predict(x_train)
    original_classes = np.argmax(y_train_orig, axis=1)
    
    # Target: class 281 (tabby cat)
    target_class = 281
    y_target = np.zeros((num_train, 1000))
    y_target[:, target_class] = 1
    
    print(f"Original classifications: {original_classes[:5]}")
    print(f"Target class: {target_class}")
    
    # 4. Create and train patch
    attack = AdversarialPatchPyTorch(
        estimator=classifier,
        rotation_max=30.0,
        scale_min=0.2,
        scale_max=0.5,
        learning_rate=1.0,
        max_iter=200,
        batch_size=8,
        patch_shape=(3, 100, 100),
        patch_type='circle',
        targeted=True,
        verbose=True
    )
    
    print("\nTraining adversarial patch...")
    patch, patch_mask = attack.generate(x=x_train, y=y_target)
    
    # 5. Test on new images
    x_test = np.random.rand(10, 3, 224, 224).astype(np.float32)
    
    # Get predictions without patch
    pred_clean = np.argmax(classifier.predict(x_test), axis=1)
    
    # Apply patch and get predictions
    x_test_patched = attack.apply_patch(x=x_test, scale=0.4)
    pred_patched = np.argmax(classifier.predict(x_test_patched), axis=1)
    
    # 6. Calculate success rate
    success_rate = np.sum(pred_patched == target_class) / len(pred_patched)
    
    print(f"\nResults:")
    print(f"Clean predictions: {pred_clean}")
    print(f"Patched predictions: {pred_patched}")
    print(f"Attack success rate: {success_rate * 100:.2f}%")
    print(f"Target class {target_class} achieved: {np.sum(pred_patched == target_class)}/{len(pred_patched)} times")
    
    return patch, patch_mask, attack

# Run the complete example
patch, mask, attack = create_adversarial_patch_complete_example()