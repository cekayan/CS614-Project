#!/usr/bin/env python3
"""
Optimized BreakHis Transfer Learning - Individual Model Results
=============================================================

Focus:
1. Clear individual model results on train/validation/test
2. Configurable number of trainable CNN layers
3. Advanced training techniques for better performance
4. No ensemble - just clean individual results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import copy
import time
import numpy as np
from breakhis_transfer_trainer import (
    download_breakhis_dataset, create_breakhis_dataset_from_raw, 
    BreakHisDataset, count_trainable_parameters
)
from dataloader import create_dataloaders
from sklearn.metrics import precision_recall_fscore_support

class ConfigurableResNet(nn.Module):
    """
    ResNet50 with configurable number of trainable layers.
    """
    def __init__(self, num_classes=2, trainable_layers=2):
        super(ConfigurableResNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        
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

class ConfigurableMobileNet(nn.Module):
    """
    MobileNetV2 with configurable number of trainable feature blocks.
    """
    def __init__(self, num_classes=2, trainable_blocks=2):
        super(ConfigurableMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V2')
        
        # Freeze all layers first
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
        # MobileNetV2 has 19 feature blocks (0-18)
        # Unfreeze the last N blocks
        total_blocks = 19
        start_block = max(0, total_blocks - trainable_blocks)
        
        trainable_block_nums = list(range(start_block, total_blocks))
        for block_num in trainable_block_nums:
            for param in self.mobilenet.features[block_num].parameters():
                param.requires_grad = True
                
        print(f"  → Trainable feature blocks: {trainable_block_nums}")
        
        # Advanced classifier
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        for param in self.mobilenet.classifier.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.mobilenet(x)

def create_optimized_augmentations():
    """Create effective data augmentations without TTA complications."""
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def train_optimized_model(model, dataloaders, dataset_sizes, device, num_epochs=15):
    """
    Optimized training with advanced techniques.
    """
    # Advanced optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Separate learning rates
    classifier_params = []
    feature_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                feature_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': feature_params, 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': classifier_params, 'lr': 1e-3, 'weight_decay': 1e-3}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-6)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'  Epoch {epoch+1}/{num_epochs}')
        print('  ' + '-' * 40)
        
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
            
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'  {phase.capitalize()}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Save best model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    print(f'  Best validation accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_all_splits(model, dataloaders, dataset_sizes, device, model_name):
    """
    Evaluate model on all splits and return comprehensive results.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    print(f"\n📊 {model_name} - Evaluation Results:")
    print("-" * 60)
    print(f"{'Split':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Loss':<10}")
    print("-" * 60)
    
    for split in ['train', 'validation', 'test']:
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloaders[split]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        loss_avg = running_loss / dataset_sizes[split]
        acc = running_corrects.double() / dataset_sizes[split]
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        results[split] = {
            'accuracy': acc.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': loss_avg
        }
        
        print(f"{split.capitalize():<12} {acc.item():<10.4f} {precision:<12.4f} {recall:<10.4f} {f1:<10.4f} {loss_avg:<10.4f}")
    
    return results

def experiment_with_layers(all_images, osteo_dataloaders, osteo_sizes, osteo_classes, device):
    """
    Experiment with different numbers of trainable layers.
    """
    # Create BreakHis dataloaders
    train_transforms, val_test_transforms = create_optimized_augmentations()
    
    import random
    all_paths = []
    all_labels = []
    label_map = {'benign': 0, 'malignant': 1}
    
    for class_name, paths in all_images.items():
        all_paths.extend(paths)
        all_labels.extend([label_map[class_name]] * len(paths))
    
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    total_samples = len(all_paths)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)
    
    train_dataset = BreakHisDataset(all_paths[:train_end], all_labels[:train_end], train_transforms)
    val_dataset = BreakHisDataset(all_paths[train_end:val_end], all_labels[train_end:val_end], val_test_transforms)
    test_dataset = BreakHisDataset(all_paths[val_end:], all_labels[val_end:], val_test_transforms)
    
    breakhis_dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
        'validation': torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    }
    
    breakhis_sizes = {'train': len(train_dataset), 'validation': len(val_dataset), 'test': len(test_dataset)}
    
    # Configuration for experiments
    experiments = [
        ("ResNet50 - 1 Layer", ConfigurableResNet, {"trainable_layers": 1}),
        ("ResNet50 - 2 Layers", ConfigurableResNet, {"trainable_layers": 2}),
        ("ResNet50 - 3 Layers", ConfigurableResNet, {"trainable_layers": 3}),
        ("MobileNetV2 - 2 Blocks", ConfigurableMobileNet, {"trainable_blocks": 2}),
        ("MobileNetV2 - 3 Blocks", ConfigurableMobileNet, {"trainable_blocks": 3}),
        ("MobileNetV2 - 4 Blocks", ConfigurableMobileNet, {"trainable_blocks": 4}),
    ]
    
    all_results = {}
    
    for exp_name, model_class, model_kwargs in experiments:
        print(f"\n{'='*70}")
        print(f"🔥 EXPERIMENT: {exp_name}")
        print(f"{'='*70}")
        
        # Step 1: Train on BreakHis
        print(f"\n📚 Phase 1: Training on BreakHis")
        model = model_class(num_classes=2, **model_kwargs).to(device)
        
        total_params, trainable_params = count_trainable_parameters(model)
        print(f"  Parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}% trainable)")
        
        model, history = train_optimized_model(model, breakhis_dataloaders, breakhis_sizes, device, num_epochs=15)
        
        # Step 2: Transfer to osteosarcoma
        print(f"\n🔄 Phase 2: Transfer to Osteosarcoma")
        
        # Freeze BreakHis-trained features
        for name, param in model.named_parameters():
            if param.requires_grad and ('layer' in name or 'features' in name):
                param.requires_grad = False
        
        # Replace classifier for 3 classes
        if hasattr(model, 'resnet'):
            model.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, len(osteo_classes))
            ).to(device)
        else:
            model.mobilenet.classifier = nn.Sequential(
                nn.BatchNorm1d(1280),
                nn.Dropout(0.3),
                nn.Linear(1280, len(osteo_classes))
            ).to(device)
        
        # Fine-tune on osteosarcoma
        model, _ = train_optimized_model(model, osteo_dataloaders, osteo_sizes, device, num_epochs=8)
        
        # Step 3: Evaluate on all splits
        results = evaluate_all_splits(model, osteo_dataloaders, osteo_sizes, device, exp_name)
        all_results[exp_name] = results
        
        # Save model
        torch.save(model.state_dict(), f'./optimized_{exp_name.replace(" ", "_").replace("-", "").lower()}_model.pth')
    
    return all_results

def print_summary_comparison(all_results):
    """
    Print a comparison summary of all experiments.
    """
    print(f"\n{'='*80}")
    print("🏆 SUMMARY COMPARISON - TEST SET RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Experiment':<25} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    best_acc = 0
    best_model = ""
    
    for exp_name, results in all_results.items():
        test_results = results['test']
        acc = test_results['accuracy']
        
        if acc > best_acc:
            best_acc = acc
            best_model = exp_name
        
        print(f"{exp_name:<25} {acc:<10.4f} {test_results['precision']:<12.4f} {test_results['recall']:<10.4f} {test_results['f1']:<10.4f}")
    
    print("-" * 80)
    print(f"🥇 Best Model: {best_model} (Accuracy: {best_acc:.4f})")
    
    if best_acc >= 0.95:
        print("🎉 TARGET ACHIEVED: 95%+ accuracy!")
    else:
        print(f"📈 Progress: {best_acc*100:.1f}% (Target: 95%)")

def main():
    """
    Main experimental pipeline.
    """
    print("🚀 Optimized Transfer Learning - Layer Experimentation")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n📥 Loading BreakHis dataset...")
    raw_dataset_path = download_breakhis_dataset()
    if raw_dataset_path is None:
        return
    
    all_images, _, _ = create_breakhis_dataset_from_raw(raw_dataset_path)
    if all_images is None:
        return
    
    print("\n📥 Loading osteosarcoma dataset...")
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=16)
    
    print(f"BreakHis: {len(all_images['benign'])} benign + {len(all_images['malignant'])} malignant")
    print(f"Osteosarcoma: {osteo_classes} (sizes: {osteo_sizes})")
    
    # Run experiments
    all_results = experiment_with_layers(all_images, osteo_dataloaders, osteo_sizes, osteo_classes, device)
    
    # Print summary
    print_summary_comparison(all_results)

if __name__ == '__main__':
    main()