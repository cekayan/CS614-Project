#!/usr/bin/env python3
"""
Model Architecture Inspector
============================

This script analyzes and displays the architecture of our modified models:
1. Shows which layers are frozen vs trainable
2. Displays classifier head structure
3. Compares parameter counts
"""

import torch
import torch.nn as nn
from torchvision import models
from breakhis_transfer_trainer import BreakHisResNetClassifier, BreakHisMobileNetClassifier

def print_model_structure(model, model_name):
    """Print detailed model structure with trainable status."""
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} ARCHITECTURE ANALYSIS")
    print(f"{'='*60}")
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n📋 LAYER-BY-LAYER ANALYSIS:")
    print("-" * 80)
    print(f"{'Layer Name':<35} {'Trainable':<12} {'Parameters':<15} {'Shape'}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            
        trainable_status = "✅ YES" if param.requires_grad else "❌ NO"
        shape_str = str(list(param.shape))
        
        print(f"{name:<35} {trainable_status:<12} {param_count:<15,} {shape_str}")
    
    print("-" * 80)
    print(f"{'TOTAL':<35} {'':<12} {total_params:<15,}")
    print(f"{'TRAINABLE':<35} {'':<12} {trainable_params:<15,} ({100*trainable_params/total_params:.1f}%)")
    
    return total_params, trainable_params

def analyze_classifier_head(model, model_name):
    """Analyze the classifier head structure."""
    print(f"\n🧠 {model_name.upper()} CLASSIFIER HEAD STRUCTURE:")
    print("-" * 50)
    
    if hasattr(model, 'resnet'):
        classifier = model.resnet.fc
        input_features = 2048  # ResNet50 feature size
    elif hasattr(model, 'mobilenet'):
        classifier = model.mobilenet.classifier
        input_features = 1280  # MobileNetV2 feature size
    
    print(f"Input features: {input_features}")
    print(f"Classifier structure:")
    for i, layer in enumerate(classifier):
        print(f"  [{i}] {layer}")
    
    # Calculate output for each layer
    print(f"\nData flow through classifier:")
    current_size = input_features
    for i, layer in enumerate(classifier):
        if isinstance(layer, nn.Linear):
            print(f"  Step {i}: {current_size} → {layer.out_features} (Linear)")
            current_size = layer.out_features
        elif isinstance(layer, nn.Dropout):
            print(f"  Step {i}: {current_size} → {current_size} (Dropout p={layer.p})")
        elif isinstance(layer, nn.ReLU):
            print(f"  Step {i}: {current_size} → {current_size} (ReLU)")

def compare_with_original():
    """Compare with original pre-trained models."""
    print(f"\n{'='*60}")
    print("COMPARISON WITH ORIGINAL MODELS")
    print(f"{'='*60}")
    
    # Original ResNet50
    original_resnet = models.resnet50(weights='IMAGENET1K_V1')
    orig_resnet_params = sum(p.numel() for p in original_resnet.parameters())
    
    # Original MobileNetV2  
    original_mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
    orig_mobilenet_params = sum(p.numel() for p in original_mobilenet.parameters())
    
    print(f"\n📊 PARAMETER COMPARISON:")
    print("-" * 60)
    print(f"{'Model':<20} {'Original':<15} {'Modified':<15} {'Trainable':<15}")
    print("-" * 60)
    
    # Analyze our models
    custom_resnet = BreakHisResNetClassifier(num_classes=2)
    custom_mobilenet = BreakHisMobileNetClassifier(num_classes=2)
    
    resnet_total = sum(p.numel() for p in custom_resnet.parameters())
    resnet_trainable = sum(p.numel() for p in custom_resnet.parameters() if p.requires_grad)
    
    mobilenet_total = sum(p.numel() for p in custom_mobilenet.parameters())
    mobilenet_trainable = sum(p.numel() for p in custom_mobilenet.parameters() if p.requires_grad)
    
    print(f"{'ResNet50':<20} {orig_resnet_params:<15,} {resnet_total:<15,} {resnet_trainable:<15,}")
    print(f"{'MobileNetV2':<20} {orig_mobilenet_params:<15,} {mobilenet_total:<15,} {mobilenet_trainable:<15,}")
    
    print(f"\n🎯 TRAINABLE PERCENTAGE:")
    print(f"ResNet50: {100*resnet_trainable/resnet_total:.1f}% of parameters are trainable")
    print(f"MobileNetV2: {100*mobilenet_trainable/mobilenet_total:.1f}% of parameters are trainable")

def show_layer_groups():
    """Show which layer groups are frozen vs trainable."""
    print(f"\n{'='*60}")
    print("LAYER GROUP TRAINING STATUS")
    print(f"{'='*60}")
    
    print(f"\n🔴 RESNET50 LAYER GROUPS:")
    print("-" * 40)
    resnet_layers = [
        ("conv1 + bn1", "❌ FROZEN"),
        ("layer1 (residual blocks)", "❌ FROZEN"), 
        ("layer2 (residual blocks)", "❌ FROZEN"),
        ("layer3 (residual blocks)", "✅ TRAINABLE"),
        ("layer4 (residual blocks)", "✅ TRAINABLE"),
        ("avgpool", "❌ FROZEN"),
        ("fc (custom classifier)", "✅ TRAINABLE")
    ]
    
    for layer, status in resnet_layers:
        print(f"  {layer:<25} {status}")
    
    print(f"\n📱 MOBILENETV2 LAYER GROUPS:")
    print("-" * 40)
    mobilenet_layers = [
        ("features[0-16] (blocks)", "❌ FROZEN"),
        ("features[17] (block)", "✅ TRAINABLE"),
        ("features[18] (block)", "✅ TRAINABLE"),
        ("classifier (custom)", "✅ TRAINABLE")
    ]
    
    for layer, status in mobilenet_layers:
        print(f"  {layer:<25} {status}")

def main():
    """Main analysis function."""
    print("🔍 MODEL ARCHITECTURE INSPECTION")
    print("=" * 60)
    
    # Create models
    resnet_model = BreakHisResNetClassifier(num_classes=2)
    mobilenet_model = BreakHisMobileNetClassifier(num_classes=2)
    
    # Show layer group status
    show_layer_groups()
    
    # Analyze classifier heads
    analyze_classifier_head(resnet_model, "ResNet50")
    analyze_classifier_head(mobilenet_model, "MobileNetV2")
    
    # Compare with originals
    compare_with_original()
    
    # Detailed layer analysis (commented out as it's very long)
    print(f"\n💡 To see detailed layer-by-layer analysis, uncomment the lines below:")
    print("# print_model_structure(resnet_model, 'ResNet50')")
    print("# print_model_structure(mobilenet_model, 'MobileNetV2')")
    
    # Uncomment these lines if you want to see ALL layers:
    # print_model_structure(resnet_model, "ResNet50")
    # print_model_structure(mobilenet_model, "MobileNetV2")
    
    print(f"\n✅ Architecture analysis complete!")

if __name__ == '__main__':
    main()