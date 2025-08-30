#!/usr/bin/env python3
"""
BreakHis Transfer Learning Script
=================================

This script:
1. Downloads the BreakHis dataset from Kaggle
2. Trains only the last couple of layers of ResNet50/MobileNetV2 on BreakHis
3. Evaluates the fine-tuned models on the osteosarcoma dataset

The approach uses transfer learning where we:
- Load ImageNet pre-trained models
- Freeze most layers (feature extraction)
- Train only the classifier layers on BreakHis
- Test cross-domain transfer to osteosarcoma classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import shutil
import kagglehub
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataloader import create_dataloaders

class BreakHisResNetClassifier(nn.Module):
    """
    ResNet50 model for BreakHis dataset with only last 2 layers trainable.
    Freezes all layers except the last 2 convolutional blocks and classifier.
    """
    def __init__(self, num_classes=2):
        super(BreakHisResNetClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        
        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze only the last 2 layers (layer3 and layer4) + classifier
        # ResNet50 has: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        
        # Unfreeze layer3 (second-to-last convolutional block)
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
            
        # Unfreeze layer4 (last convolutional block)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer with our custom classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Ensure the new classifier layers are trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.resnet(x)

class BreakHisMobileNetClassifier(nn.Module):
    """
    MobileNetV2 model for BreakHis dataset with only last 2 layers trainable.
    Freezes all layers except the last 2 feature blocks and classifier.
    """
    def __init__(self, num_classes=2):
        super(BreakHisMobileNetClassifier, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Freeze all layers first
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
        # MobileNetV2 features has 19 inverted residual blocks (0-18)
        # Unfreeze only the last 2 blocks (17 and 18) + classifier
        
        # Unfreeze the last 2 feature blocks
        for param in self.mobilenet.features[17].parameters():
            param.requires_grad = True
            
        for param in self.mobilenet.features[18].parameters():
            param.requires_grad = True
        
        # Get the number of features from the last layer
        num_features = self.mobilenet.classifier[1].in_features
        
        # Replace the final classifier with our custom one
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        # Ensure the new classifier layers are trainable
        for param in self.mobilenet.classifier.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.mobilenet(x)

class BreakHisDenseNetClassifier(nn.Module):
    """
    DenseNet121 model for BreakHis dataset with only last 2 layers trainable.
    Freezes all layers except the last 2 dense blocks and classifier.
    """
    def __init__(self, num_classes=2):
        super(BreakHisDenseNetClassifier, self).__init__()
        
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(weights='IMAGENET1K_V1')
        
        # Freeze all layers first
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        # DenseNet121 has 4 dense blocks: denseblock1, denseblock2, denseblock3, denseblock4
        # Unfreeze only the last 2 blocks (denseblock3 and denseblock4) + classifier
        
        # Unfreeze denseblock3 (third dense block)
        for param in self.densenet.features.denseblock3.parameters():
            param.requires_grad = True
        for param in self.densenet.features.transition3.parameters():
            param.requires_grad = True
            
        # Unfreeze denseblock4 (last dense block) 
        for param in self.densenet.features.denseblock4.parameters():
            param.requires_grad = True
        for param in self.densenet.features.norm5.parameters():
            param.requires_grad = True
        
        # Get the number of features from the last layer
        num_features = self.densenet.classifier.in_features
        
        # Replace the final classifier with our custom one
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Ensure the new classifier layers are trainable
        for param in self.densenet.classifier.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.densenet(x)

def download_breakhis_dataset():
    """
    Download BreakHis dataset from Kaggle using kagglehub.
    """
    print("📥 Downloading BreakHis dataset from Kaggle using kagglehub...")
    
    try:
        # Download using kagglehub
        print("Downloading dataset...")
        path = kagglehub.dataset_download("ambarish/breakhis")
        
        print(f"✅ Dataset downloaded to: {path}")
        print(f"Exploring dataset structure...")
        
        # List the contents to understand the structure
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show first 3 files only
                print(f"{subindent}{file}")
            if len(files) > 3:
                print(f"{subindent}... and {len(files)-3} more files")
        
        return path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please ensure:")
        print("1. kagglehub is installed: pip install kagglehub")
        print("2. You have internet connection")
        print("3. You have access to the ambarish/breakhis dataset")
        return None

def create_breakhis_dataset_from_raw(raw_path):
    """
    Create dataset directly from the raw BreakHis structure without saving.
    BreakHis structure: BreaKHis_v1/histology_slides/breast/[benign|malignant]/
    """
    print("🔍 Creating dataset from raw BreakHis structure...")
    
    # Find the actual data path
    breakhis_root = None
    for root, dirs, files in os.walk(raw_path):
        if 'breast' in dirs:
            breakhis_root = root
            break
    
    if breakhis_root is None:
        print("❌ Could not find breast data directory in BreakHis dataset")
        return None, None, None
    
    breast_path = os.path.join(breakhis_root, 'breast')
    benign_path = os.path.join(breast_path, 'benign')
    malignant_path = os.path.join(breast_path, 'malignant')
    
    print(f"Found breast data at: {breast_path}")
    print(f"Benign path: {benign_path}")
    print(f"Malignant path: {malignant_path}")
    
    # Collect all image paths
    all_images = {'benign': [], 'malignant': []}
    
    # Collect benign images
    for root, dirs, files in os.walk(benign_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_images['benign'].append(os.path.join(root, file))
    
    # Collect malignant images  
    for root, dirs, files in os.walk(malignant_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                all_images['malignant'].append(os.path.join(root, file))
    
    print(f"Found {len(all_images['benign'])} benign images")
    print(f"Found {len(all_images['malignant'])} malignant images")
    
    return all_images, benign_path, malignant_path

class BreakHisDataset(torch.utils.data.Dataset):
    """Custom dataset for BreakHis that works directly with image paths."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_breakhis_dataloaders_from_raw(all_images, batch_size=32):
    """
    Create data loaders directly from BreakHis image paths without saving.
    """
    import random
    
    IMAGE_SIZE = 224
    
    # Data augmentation for training
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation/test
    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare data splits
    all_paths = []
    all_labels = []
    
    # Label mapping: benign=0, malignant=1
    label_map = {'benign': 0, 'malignant': 1}
    
    for class_name, paths in all_images.items():
        all_paths.extend(paths)
        all_labels.extend([label_map[class_name]] * len(paths))
    
    # Shuffle the data
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    # Split data: 70% train, 15% val, 15% test
    total_samples = len(all_paths)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)
    
    train_paths = all_paths[:train_end]
    train_labels = all_labels[:train_end]
    
    val_paths = all_paths[train_end:val_end]
    val_labels = all_labels[train_end:val_end]
    
    test_paths = all_paths[val_end:]
    test_labels = all_labels[val_end:]
    
    # Create datasets
    train_dataset = BreakHisDataset(train_paths, train_labels, train_transforms)
    val_dataset = BreakHisDataset(val_paths, val_labels, val_test_transforms)
    test_dataset = BreakHisDataset(test_paths, test_labels, val_test_transforms)
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'validation': len(val_dataset),
        'test': len(test_dataset)
    }
    
    class_names = ['benign', 'malignant']
    
    print(f"BreakHis Dataset - Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    return dataloaders, dataset_sizes, class_names

def count_trainable_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def debug_parameter_breakdown(model, model_name):
    """Debug function to show parameter breakdown by layer."""
    print(f"\n🔍 {model_name} Parameter Breakdown:")
    print("-" * 50)
    
    if hasattr(model, 'resnet'):
        base_model = model.resnet
        layer_groups = [
            ('conv1 + bn1', [base_model.conv1, base_model.bn1]),
            ('layer1', [base_model.layer1]),
            ('layer2', [base_model.layer2]), 
            ('layer3', [base_model.layer3]),
            ('layer4', [base_model.layer4]),
            ('avgpool', [base_model.avgpool]),
            ('fc', [base_model.fc])
        ]
    elif hasattr(model, 'mobilenet'):
        base_model = model.mobilenet
        layer_groups = [
            ('features[0-16]', [base_model.features[i] for i in range(17)]),
            ('features[17]', [base_model.features[17]]),
            ('features[18]', [base_model.features[18]]),
            ('classifier', [base_model.classifier])
        ]
    elif hasattr(model, 'densenet'):
        base_model = model.densenet
        layer_groups = [
            ('conv0 + norm0', [base_model.features.conv0, base_model.features.norm0]),
            ('denseblock1', [base_model.features.denseblock1, base_model.features.transition1]),
            ('denseblock2', [base_model.features.denseblock2, base_model.features.transition2]),
            ('denseblock3', [base_model.features.denseblock3, base_model.features.transition3]),
            ('denseblock4', [base_model.features.denseblock4, base_model.features.norm5]),
            ('classifier', [base_model.classifier])
        ]
    
    for layer_name, layers in layer_groups:
        total_params = 0
        trainable_params = 0
        
        for layer in layers:
            for param in layer.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        status = "✅ TRAINABLE" if trainable_params > 0 else "❌ FROZEN"
        print(f"{layer_name:<15} {total_params:>10,} params  {status}")
    
    print("-" * 50)

def train_model_frozen(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, num_epochs=10, device='cuda'):
    """
    Train model with frozen feature extraction layers.
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("🔥 Training with frozen feature extraction layers...")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_cross_domain(model, osteosarcoma_dataloaders, dataset_sizes, criterion, device='cuda'):
    """
    Evaluate the BreakHis-trained model on osteosarcoma dataset.
    """
    print("\n" + "="*60)
    print("🔬 CROSS-DOMAIN EVALUATION ON OSTEOSARCOMA DATASET")
    print("="*60)
    
    results = {}
    
    for split in ['train', 'validation', 'test']:
        print(f"\nEvaluating on osteosarcoma {split} set...")
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in osteosarcoma_dataloaders[split]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        loss = running_loss / dataset_sizes[split]
        acc = running_corrects.double() / dataset_sizes[split]
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        results[split] = {
            'loss': loss,
            'accuracy': acc.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f'{split.capitalize()} - Loss: {loss:.4f} Acc: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
    
    return results

def print_cross_domain_results(resnet_results, mobilenet_results, densenet_results):
    """
    Print comparison table for cross-domain results.
    """
    print("\n" + "="*80)
    print("🏆 CROSS-DOMAIN TRANSFER LEARNING RESULTS")
    print("="*80)
    
    print("\n📊 ResNet50 (BreakHis → Osteosarcoma)")
    print("-" * 60)
    print(f"{'Split':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for split in ['train', 'validation', 'test']:
        r = resnet_results[split]
        print(f"{split.capitalize():<12} {r['accuracy']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}")
    
    print("\n📊 MobileNetV2 (BreakHis → Osteosarcoma)")
    print("-" * 60)
    print(f"{'Split':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for split in ['train', 'validation', 'test']:
        r = mobilenet_results[split]
        print(f"{split.capitalize():<12} {r['accuracy']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}")
    
    print("\n📊 DenseNet121 (BreakHis → Osteosarcoma)")
    print("-" * 60)
    print(f"{'Split':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for split in ['train', 'validation', 'test']:
        r = densenet_results[split]
        print(f"{split.capitalize():<12} {r['accuracy']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}")

def main():
    """
    Main pipeline for BreakHis transfer learning experiment.
    """
    print("🔬 BreakHis → Osteosarcoma Transfer Learning Experiment")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Download BreakHis dataset and create dataloaders
    print("\n" + "="*50)
    print("STEP 1: DATASET PREPARATION")
    print("="*50)
    
    print("📥 Downloading BreakHis dataset...")
    # Download dataset using kagglehub
    raw_dataset_path = download_breakhis_dataset()
    
    if raw_dataset_path is None:
        print("❌ Failed to download BreakHis dataset")
        return
    
    # Create dataset directly from raw structure
    all_images, benign_path, malignant_path = create_breakhis_dataset_from_raw(raw_dataset_path)
    
    if all_images is None:
        print("❌ Failed to parse BreakHis dataset structure")
        return
    
    # Create dataloaders directly from image paths
    breakhis_dataloaders, breakhis_sizes, breakhis_classes = create_breakhis_dataloaders_from_raw(all_images, batch_size=32)
    
    if breakhis_dataloaders is None:
        print("❌ Failed to create BreakHis dataloaders")
        return
    
    # Load osteosarcoma dataset for evaluation
    print("\nLoading osteosarcoma dataset for evaluation...")
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=32)
    
    print(f"BreakHis classes: {breakhis_classes} (samples: {breakhis_sizes})")
    print(f"Osteosarcoma classes: {osteo_classes} (samples: {osteo_sizes})")
    
    # Step 2: Train ResNet50 on BreakHis
    print("\n" + "="*50)
    print("STEP 2: TRAINING RESNET50 ON BREAKHIS")
    print("="*50)
    
    resnet_model = BreakHisResNetClassifier(num_classes=len(breakhis_classes))
    resnet_model = resnet_model.to(device)
    
    # Display parameter info
    total_params, trainable_params = count_trainable_parameters(resnet_model)
    print(f"ResNet50 - Total parameters: {total_params:,}")
    print(f"ResNet50 - Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    debug_parameter_breakdown(resnet_model, "ResNet50")
    
    # Train the last 2 conv layers + classifier
    criterion = nn.CrossEntropyLoss()
    # Get all trainable parameters (layer3, layer4, and fc)
    trainable_param_list = [p for p in resnet_model.parameters() if p.requires_grad]
    resnet_optimizer = optim.Adam(trainable_param_list, lr=1e-4)  # Lower LR for conv layers
    resnet_scheduler = optim.lr_scheduler.StepLR(resnet_optimizer, step_size=5, gamma=0.5)
    
    resnet_model, resnet_history = train_model_frozen(
        resnet_model, breakhis_dataloaders, breakhis_sizes,
        criterion, resnet_optimizer, resnet_scheduler, num_epochs=10, device=device
    )
    
    # Step 3: Train MobileNetV2 on BreakHis
    print("\n" + "="*50)
    print("STEP 3: TRAINING MOBILENETV2 ON BREAKHIS")
    print("="*50)
    
    mobilenet_model = BreakHisMobileNetClassifier(num_classes=len(breakhis_classes))
    mobilenet_model = mobilenet_model.to(device)
    
    # Display parameter info
    total_params, trainable_params = count_trainable_parameters(mobilenet_model)
    print(f"MobileNetV2 - Total parameters: {total_params:,}")
    print(f"MobileNetV2 - Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    debug_parameter_breakdown(mobilenet_model, "MobileNetV2")
    
    # Train the last 2 feature blocks + classifier
    trainable_param_list = [p for p in mobilenet_model.parameters() if p.requires_grad]
    mobilenet_optimizer = optim.Adam(trainable_param_list, lr=1e-4)  # Lower LR for conv layers
    mobilenet_scheduler = optim.lr_scheduler.StepLR(mobilenet_optimizer, step_size=5, gamma=0.5)
    
    mobilenet_model, mobilenet_history = train_model_frozen(
        mobilenet_model, breakhis_dataloaders, breakhis_sizes,
        criterion, mobilenet_optimizer, mobilenet_scheduler, num_epochs=10, device=device
    )
    
    # Step 3.5: Train DenseNet121 on BreakHis
    print("\n" + "="*50)
    print("STEP 3.5: TRAINING DENSENET121 ON BREAKHIS")
    print("="*50)
    
    densenet_model = BreakHisDenseNetClassifier(num_classes=len(breakhis_classes))
    densenet_model = densenet_model.to(device)
    
    # Display parameter info
    total_params, trainable_params = count_trainable_parameters(densenet_model)
    print(f"DenseNet121 - Total parameters: {total_params:,}")
    print(f"DenseNet121 - Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    debug_parameter_breakdown(densenet_model, "DenseNet121")
    
    # Train the last 2 dense blocks + classifier
    trainable_param_list = [p for p in densenet_model.parameters() if p.requires_grad]
    densenet_optimizer = optim.Adam(trainable_param_list, lr=1e-4)  # Lower LR for conv layers
    densenet_scheduler = optim.lr_scheduler.StepLR(densenet_optimizer, step_size=5, gamma=0.5)
    
    densenet_model, densenet_history = train_model_frozen(
        densenet_model, breakhis_dataloaders, breakhis_sizes,
        criterion, densenet_optimizer, densenet_scheduler, num_epochs=10, device=device
    )
    
    # Step 4: Transfer learning on osteosarcoma
    print("\n" + "="*50)
    print("STEP 4: TRANSFER LEARNING ON OSTEOSARCOMA")
    print("="*50)
    
    print("🔄 Adapting BreakHis-trained models for osteosarcoma classification...")
    
    # === ResNet50 Transfer ===
    print("\n--- ResNet50 Transfer Learning ---")
    
    # Freeze the BreakHis-trained CNN layers (preserve learned features)
    for param in resnet_model.resnet.layer3.parameters():
        param.requires_grad = False
    for param in resnet_model.resnet.layer4.parameters():
        param.requires_grad = False
    
    # Replace only the classifier for 3-class osteosarcoma
    resnet_model.resnet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, len(osteo_classes))
    ).to(device)
    
    # Train only the new classifier
    resnet_osteo_optimizer = optim.Adam(resnet_model.resnet.fc.parameters(), lr=1e-3)
    resnet_osteo_scheduler = optim.lr_scheduler.StepLR(resnet_osteo_optimizer, step_size=3, gamma=0.5)
    
    print("Training ResNet50 classifier on osteosarcoma...")
    resnet_model, _ = train_model_frozen(
        resnet_model, osteo_dataloaders, osteo_sizes,
        criterion, resnet_osteo_optimizer, resnet_osteo_scheduler, num_epochs=5, device=device
    )
    
    # === MobileNetV2 Transfer ===
    print("\n--- MobileNetV2 Transfer Learning ---")
    
    # Freeze the BreakHis-trained CNN layers
    for param in mobilenet_model.mobilenet.features[17].parameters():
        param.requires_grad = False
    for param in mobilenet_model.mobilenet.features[18].parameters():
        param.requires_grad = False
    
    # Replace only the classifier for 3-class osteosarcoma
    mobilenet_model.mobilenet.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, len(osteo_classes))
    ).to(device)
    
    # Train only the new classifier
    mobilenet_osteo_optimizer = optim.Adam(mobilenet_model.mobilenet.classifier.parameters(), lr=1e-3)
    mobilenet_osteo_scheduler = optim.lr_scheduler.StepLR(mobilenet_osteo_optimizer, step_size=3, gamma=0.5)
    
    print("Training MobileNetV2 classifier on osteosarcoma...")
    mobilenet_model, _ = train_model_frozen(
        mobilenet_model, osteo_dataloaders, osteo_sizes,
        criterion, mobilenet_osteo_optimizer, mobilenet_osteo_scheduler, num_epochs=5, device=device
    )
    
    # === DenseNet121 Transfer ===
    print("\n--- DenseNet121 Transfer Learning ---")
    
    # Freeze the BreakHis-trained CNN layers
    for param in densenet_model.densenet.features.denseblock3.parameters():
        param.requires_grad = False
    for param in densenet_model.densenet.features.transition3.parameters():
        param.requires_grad = False
    for param in densenet_model.densenet.features.denseblock4.parameters():
        param.requires_grad = False
    for param in densenet_model.densenet.features.norm5.parameters():
        param.requires_grad = False
    
    # Replace only the classifier for 3-class osteosarcoma
    densenet_model.densenet.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1024, len(osteo_classes))
    ).to(device)
    
    # Train only the new classifier
    densenet_osteo_optimizer = optim.Adam(densenet_model.densenet.classifier.parameters(), lr=1e-3)
    densenet_osteo_scheduler = optim.lr_scheduler.StepLR(densenet_osteo_optimizer, step_size=3, gamma=0.5)
    
    print("Training DenseNet121 classifier on osteosarcoma...")
    densenet_model, _ = train_model_frozen(
        densenet_model, osteo_dataloaders, osteo_sizes,
        criterion, densenet_osteo_optimizer, densenet_osteo_scheduler, num_epochs=5, device=device
    )
    
    # Step 5: Final evaluation
    print("\n" + "="*50)
    print("STEP 5: FINAL EVALUATION")
    print("="*50)
    
    # Evaluate all three models
    resnet_results = evaluate_cross_domain(resnet_model, osteo_dataloaders, osteo_sizes, criterion, device)
    mobilenet_results = evaluate_cross_domain(mobilenet_model, osteo_dataloaders, osteo_sizes, criterion, device)
    densenet_results = evaluate_cross_domain(densenet_model, osteo_dataloaders, osteo_sizes, criterion, device)
    
    # Print results
    print_cross_domain_results(resnet_results, mobilenet_results, densenet_results)
    
    # Save models
    torch.save(resnet_model.state_dict(), './breakhis_resnet50_model.pth')
    torch.save(mobilenet_model.state_dict(), './breakhis_mobilenet_model.pth')
    torch.save(densenet_model.state_dict(), './breakhis_densenet121_model.pth')
    
    print("\n✅ Transfer learning experiment completed!")
    print("Models saved as:")
    print("- 'breakhis_resnet50_model.pth'")
    print("- 'breakhis_mobilenet_model.pth'") 
    print("- 'breakhis_densenet121_model.pth'")

if __name__ == '__main__':
    main()