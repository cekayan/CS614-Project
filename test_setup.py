#!/usr/bin/env python3
"""
Quick test script to verify data loading and model setup before full training.
"""

import torch
import sys
import os
#sys.path.append('/home/cek99/CS614')

from dataloader import create_dataloaders
from resnet_trainer import ResNetClassifier

def test_data_loading():
    """Test if data can be loaded successfully."""
    print("Testing data loading...")
    
    osteosarcoma_data_dir = './osteosarcoma_organized'
    
    if not os.path.exists(osteosarcoma_data_dir):
        print(f"❌ Data directory not found: {osteosarcoma_data_dir}")
        return False
    
    try:
        dataloaders, dataset_sizes, class_names = create_dataloaders(osteosarcoma_data_dir, batch_size=4)
        print(f"✅ Data loaded successfully!")
        print(f"   Classes: {class_names}")
        print(f"   Dataset sizes: {dataset_sizes}")
        
        # Test loading one batch
        inputs, labels = next(iter(dataloaders['train']))
        print(f"   Sample batch shape: {inputs.shape}, labels: {labels.shape}")
        return True, dataloaders, dataset_sizes, class_names
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_model_creation(num_classes):
    """Test if model can be created and run forward pass."""
    print("\nTesting model creation...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        model = ResNetClassifier(num_classes=num_classes)
        model = model.to(device)
        print(f"✅ Model created successfully!")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Forward pass successful! Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def main():
    print("🔍 Testing ResNet50 Transfer Learning Setup")
    print("=" * 50)
    
    # Test data loading
    data_result = test_data_loading()
    if not data_result:
        print("\n❌ Data loading failed. Please check your data directory.")
        return
    
    success, dataloaders, dataset_sizes, class_names = data_result
    
    # Test model creation
    model_success = test_model_creation(len(class_names))
    if not model_success:
        print("\n❌ Model creation failed.")
        return
    
    print("\n🎉 All tests passed! Ready to start training.")
    print("\nTo start training, run:")
    print("   python /home/cek99/CS614/resnet_trainer.py")
    
    # Show what training will do
    print(f"\nTraining will:")
    print(f"   • Use {len(class_names)} classes: {class_names}")
    print(f"   • Train on {dataset_sizes['train']} samples")
    print(f"   • Validate on {dataset_sizes['validation']} samples") 
    print(f"   • Test on {dataset_sizes['test']} samples")
    print(f"   • Use pre-trained ResNet50 with transfer learning")
    print(f"   • Run for 15 epochs with learning rate scheduling")

if __name__ == '__main__':
    main()