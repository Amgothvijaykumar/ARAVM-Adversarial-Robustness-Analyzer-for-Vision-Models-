import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from art.estimators.classification import PyTorchClassifier

def create_art_classifier(model_name='resnet18', num_classes=10, pretrained=True):
    """
    Create an ART PyTorchClassifier with various torchvision models.
    
    Supported models: resnet18, resnet50, vgg16, mobilenet_v2, efficientnet_b0, etc.
    """
    
    # Load the pre-trained model
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
        channels_first=True,
    )
    
    return classifier

# Usage
classifier = create_art_classifier(model_name='resnet50', num_classes=10, pretrained=True)