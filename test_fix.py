#!/usr/bin/env python3
"""
Quick test to verify the classifier replacement fix works correctly.
"""

import torch
import torch.nn as nn
from torchvision import models

class ConfigurableStainNormalizedResNet(nn.Module):
    """Test version of the ResNet class to verify fix."""
    def __init__(self, num_classes=2, trainable_layers=2, enable_feature_alignment=True):
        super(ConfigurableStainNormalizedResNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.enable_feature_alignment = enable_feature_alignment
        
        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the specified number of last layers
        layer_names = ['layer4', 'layer3', 'layer2', 'layer1']
        
        for i in range(min(trainable_layers, len(layer_names))):
            layer = getattr(self.resnet, layer_names[i])
            for param in layer.parameters():
                param.requires_grad = True
                
        print(f"  → Trainable CNN layers: {layer_names[:trainable_layers]}")
            
        # Advanced classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet(x)

def test_classifier_replacement():
    """Test the classifier replacement logic."""
    print("🧪 Testing classifier replacement fix...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 1: Create model for BreakHis (2 classes)
    print("\n1. Creating BreakHis model (2 classes)...")
    model = ConfigurableStainNormalizedResNet(num_classes=2, trainable_layers=2).to(device)
    
    # Print original classifier structure
    print("Original classifier structure:")
    for i, layer in enumerate(model.resnet.fc):
        print(f"  {i}: {layer}")
    
    # Step 2: Replace classifier for Osteosarcoma (3 classes)
    print("\n2. Replacing classifier for Osteosarcoma (3 classes)...")
    
    # Get the backbone features (before the classifier)
    num_features = model.resnet.fc[2].in_features  # Should be 2048 for ResNet50
    print(f"Backbone features: {num_features}")
    
    model.resnet.fc = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, 3)  # 3 classes for Osteosarcoma
    ).to(device)
    
    # Print new classifier structure
    print("New classifier structure:")
    for i, layer in enumerate(model.resnet.fc):
        print(f"  {i}: {layer}")
    
    # Step 3: Test forward pass
    print("\n3. Testing forward pass...")
    test_input = torch.randn(2, 3, 224, 224).to(device)
    
    try:
        output = model(test_input)
        print(f"✅ Success! Output shape: {output.shape}")
        print(f"Expected: [2, 3] (batch_size=2, num_classes=3)")
        
        if output.shape == (2, 3):
            print("🎉 Test PASSED - Classifier replacement works correctly!")
            return True
        else:
            print(f"❌ Test FAILED - Wrong output shape: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Test FAILED - Error during forward pass: {e}")
        return False

if __name__ == "__main__":
    success = test_classifier_replacement()
    if success:
        print("\n✅ Fix verified! The comprehensive script should work now.")
    else:
        print("\n❌ Fix needs more work.")
