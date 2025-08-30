#!/usr/bin/env python3
"""
Comprehensive Stain-Normalized Transfer Learning Script
=====================================================

This script combines:
1. Stain normalization (Reinhard/Macenko/Vahadane) 
2. Advanced training techniques from optimized_transfer_trainer.py
3. Configurable number of trainable layers
4. Multiple architecture experiments
5. Feature-level domain alignment

The ULTIMATE transfer learning script for BreakHis → Osteosarcoma!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import copy
import time
import numpy as np
from collections import defaultdict
import random

# Import our components
from stain_normalization import (
    ReinhardNormalizer, MacenkoNormalizer, VahadaneNormalizer
)
from breakhis_transfer_trainer import (
    download_breakhis_dataset, create_breakhis_dataset_from_raw, 
    BreakHisDataset, count_trainable_parameters
)
from dataloader import create_dataloaders
from full_cnn_transfer_trainer import (
    compute_domain_statistics, apply_feature_normalization
)
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class StainNormalizedDataset(Dataset):
    """Dataset wrapper that applies stain normalization on-the-fly."""
    
    def __init__(self, base_dataset, stain_normalizer=None, apply_stain_norm=True):
        self.base_dataset = base_dataset
        self.stain_normalizer = stain_normalizer
        self.apply_stain_norm = apply_stain_norm
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.apply_stain_norm and self.stain_normalizer is not None:
            try:
                if torch.is_tensor(image) and image.max() > 1.0:
                    image = image / 255.0
                
                normalized_image = self.stain_normalizer.normalize(image)
                
                if not torch.is_tensor(normalized_image):
                    normalized_image = torch.from_numpy(normalized_image)
                
                image = normalized_image
                
            except Exception as e:
                print(f"Warning: Stain normalization failed for sample {idx}: {e}")
        
        return image, label

class ConfigurableStainNormalizedResNet(nn.Module):
    """
    ResNet50 with configurable trainable layers + stain normalization support.
    Combines the best of optimized_transfer_trainer.py with stain normalization.
    """
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
            
        # Advanced classifier (from optimized_transfer_trainer.py)
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
        
        # Feature alignment buffers
        if enable_feature_alignment:
            self.register_buffer('source_mean', torch.zeros(num_features))
            self.register_buffer('source_std', torch.ones(num_features))
            self.register_buffer('target_mean', torch.zeros(num_features))
            self.register_buffer('target_std', torch.ones(num_features))
    
    def set_domain_statistics(self, source_stats, target_stats):
        """Set statistics for feature-level domain alignment."""
        if self.enable_feature_alignment:
            source_mean, source_std = source_stats
            target_mean, target_std = target_stats
            
            self.source_mean.copy_(source_mean)
            self.source_std.copy_(source_std)
            self.target_mean.copy_(target_mean)
            self.target_std.copy_(target_std)
    
    def get_features(self, x):
        """Extract features without classification."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x, apply_feature_alignment=False):
        """Forward pass with optional feature alignment."""
        if apply_feature_alignment and self.enable_feature_alignment:
            # Extract features
            features = self.get_features(x)
            
            # Apply feature-level domain alignment
            features = apply_feature_normalization(
                features, (self.source_mean, self.source_std),
                (self.target_mean, self.target_std)
            )
            
            # Classification
            return self.resnet.fc(features)
        else:
            return self.resnet(x)

class ConfigurableStainNormalizedMobileNet(nn.Module):
    """
    MobileNetV2 with configurable trainable blocks + stain normalization support.
    """
    def __init__(self, num_classes=2, trainable_blocks=2, enable_feature_alignment=True):
        super(ConfigurableStainNormalizedMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V2')
        self.enable_feature_alignment = enable_feature_alignment
        
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
        
        # Advanced classifier (from optimized_transfer_trainer.py)
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
        
        # Feature alignment buffers
        if enable_feature_alignment:
            self.register_buffer('source_mean', torch.zeros(num_features))
            self.register_buffer('source_std', torch.ones(num_features))
            self.register_buffer('target_mean', torch.zeros(num_features))
            self.register_buffer('target_std', torch.ones(num_features))
    
    def set_domain_statistics(self, source_stats, target_stats):
        """Set statistics for feature-level domain alignment."""
        if self.enable_feature_alignment:
            source_mean, source_std = source_stats
            target_mean, target_std = target_stats
            
            self.source_mean.copy_(source_mean)
            self.source_std.copy_(source_std)
            self.target_mean.copy_(target_mean)
            self.target_std.copy_(target_std)
    
    def get_features(self, x):
        """Extract features without classification."""
        features = self.mobilenet.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features
    
    def forward(self, x, apply_feature_alignment=False):
        """Forward pass with optional feature alignment."""
        if apply_feature_alignment and self.enable_feature_alignment:
            # Extract features
            features = self.get_features(x)
            
            # Apply feature-level domain alignment
            features = apply_feature_normalization(
                features, (self.source_mean, self.source_std),
                (self.target_mean, self.target_std)
            )
            
            # Classification
            return self.mobilenet.classifier(features)
        else:
            return self.mobilenet(x)

def create_optimized_augmentations():
    """Create effective data augmentations (from optimized_transfer_trainer.py)."""
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

def fit_stain_normalizer_from_datasets(source_dataset, target_dataset, method='reinhard'):
    """Fit stain normalizer using sample images from both datasets."""
    print(f"  Fitting {method} stain normalizer...")
    
    # Get a representative target image
    target_image, _ = target_dataset[0]
    
    # Create and fit normalizer
    if method.lower() == 'reinhard':
        normalizer = ReinhardNormalizer()
    elif method.lower() == 'macenko':
        normalizer = MacenkoNormalizer()
    elif method.lower() == 'vahadane':
        normalizer = VahadaneNormalizer()
    else:
        raise ValueError(f"Unknown stain normalization method: {method}")
    
    normalizer.fit(target_image)
    return normalizer

def create_stain_normalized_dataloaders(base_dataloaders, stain_normalizer, 
                                       normalize_splits=['train', 'validation', 'test']):
    """Create stain-normalized versions of dataloaders."""
    normalized_dataloaders = {}
    
    for split, dataloader in base_dataloaders.items():
        apply_norm = split in normalize_splits
        
        # Wrap dataset with stain normalization
        normalized_dataset = StainNormalizedDataset(
            dataloader.dataset, stain_normalizer, apply_norm
        )
        
        # Create new dataloader
        normalized_dataloaders[split] = DataLoader(
            normalized_dataset,
            batch_size=dataloader.batch_size,
            shuffle=(split == 'train'),
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )
    
    return normalized_dataloaders

def train_optimized_model(model, dataloaders, dataset_sizes, device, num_epochs=15, 
                         apply_feature_alignment=False):
    """
    Optimized training with advanced techniques (from optimized_transfer_trainer.py).
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
    
    for epoch in range(num_epochs):
        print(f'  Epoch {epoch+1}/{num_epochs}')
        
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
                    outputs = model(inputs, apply_feature_alignment=apply_feature_alignment)
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
            
            # Save best model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f'  Best validation accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def evaluate_all_splits(model, dataloaders, dataset_sizes, device, model_name, 
                       apply_feature_alignment=False):
    """Evaluate model on all splits and return comprehensive results."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    print(f"\n📊 {model_name} - Evaluation Results:")
    print("-" * 70)
    print(f"{'Split':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Loss':<10}")
    print("-" * 70)
    
    for split in ['train', 'validation', 'test']:
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloaders[split]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, apply_feature_alignment=apply_feature_alignment)
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

def main():
    """Main comprehensive experimental pipeline."""
    print("🚀 COMPREHENSIVE STAIN-NORMALIZED TRANSFER LEARNING")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    stain_methods = ['reinhard', 'macenko', 'vahadane', 'none']
    experiments = [
        ("ResNet50 - 1 Layer", ConfigurableStainNormalizedResNet, {"trainable_layers": 1}),
        ("ResNet50 - 2 Layers", ConfigurableStainNormalizedResNet, {"trainable_layers": 2}),
        ("ResNet50 - 3 Layers", ConfigurableStainNormalizedResNet, {"trainable_layers": 3}),
        ("MobileNetV2 - 2 Blocks", ConfigurableStainNormalizedMobileNet, {"trainable_blocks": 2}),
        ("MobileNetV2 - 3 Blocks", ConfigurableStainNormalizedMobileNet, {"trainable_blocks": 3}),
    ]
    
    all_results = defaultdict(dict)
    
    # Load datasets
    print("\n📥 Loading datasets...")
    raw_dataset_path = download_breakhis_dataset()
    if raw_dataset_path is None:
        return
    
    all_images, _, _ = create_breakhis_dataset_from_raw(raw_dataset_path)
    if all_images is None:
        return
    
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=16)
    
    # Create BreakHis dataloaders
    train_transforms, val_test_transforms = create_optimized_augmentations()
    
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
    
    base_breakhis_dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
        'validation': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    }
    
    breakhis_sizes = {'train': len(train_dataset), 'validation': len(val_dataset), 'test': len(test_dataset)}
    
    print(f"BreakHis: {len(all_images['benign'])} benign + {len(all_images['malignant'])} malignant")
    print(f"Osteosarcoma: {osteo_classes} (sizes: {osteo_sizes})")
    
    # Run comprehensive experiments
    for stain_method in stain_methods:
        for exp_name, model_class, model_kwargs in experiments:
            
            config_name = f"{stain_method}_{exp_name.replace(' ', '_').replace('-', '').lower()}"
            print(f"\n{'='*80}")
            print(f"🧪 EXPERIMENT: {stain_method.upper()} + {exp_name}")
            print(f"{'='*80}")
            
            try:
                # Create stain normalizer if needed
                if stain_method != 'none':
                    stain_normalizer = fit_stain_normalizer_from_datasets(
                        train_dataset, osteo_dataloaders['train'].dataset, method=stain_method
                    )
                    breakhis_dataloaders = create_stain_normalized_dataloaders(
                        base_breakhis_dataloaders, stain_normalizer
                    )
                else:
                    breakhis_dataloaders = base_breakhis_dataloaders
                    stain_normalizer = None
                
                # Phase 1: Train on BreakHis
                print(f"\n📚 Phase 1: Training on BreakHis")
                model = model_class(num_classes=2, **model_kwargs).to(device)
                
                total_params, trainable_params = count_trainable_parameters(model)
                print(f"  Parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}% trainable)")
                
                model = train_optimized_model(model, breakhis_dataloaders, breakhis_sizes, device, num_epochs=10)
                
                # Phase 2: Transfer to Osteosarcoma
                print(f"\n🔄 Phase 2: Transfer to Osteosarcoma")
                
                # Set up feature alignment BEFORE modifying classifier
                if hasattr(model, 'enable_feature_alignment') and model.enable_feature_alignment:
                    print("  Setting up feature-level domain alignment...")
                    
                    # Create temporary feature extractor with SAME architecture
                    temp_model = model_class(num_classes=2, **model_kwargs).to(device)
                    temp_model.load_state_dict(model.state_dict(), strict=True)  # Should match exactly
                    temp_model.eval()
                    
                    # Compute domain statistics
                    source_stats = compute_domain_statistics(breakhis_dataloaders['train'], temp_model, device)
                    target_stats = compute_domain_statistics(osteo_dataloaders['train'], temp_model, device)
                    model.set_domain_statistics(source_stats, target_stats)
                
                # NOW freeze BreakHis-trained features
                for name, param in model.named_parameters():
                    if param.requires_grad and ('layer' in name or 'features' in name):
                        param.requires_grad = False
                
                # Replace classifier for 3 classes (Osteosarcoma)
                if hasattr(model, 'resnet'):
                    # Get the backbone features (before the classifier)
                    num_features = model.resnet.fc[2].in_features  # Should be 2048 for ResNet50
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
                        nn.Linear(256, len(osteo_classes))
                    ).to(device)
                else:
                    # MobileNet
                    num_features = model.mobilenet.classifier[2].in_features  # Should be 1280 for MobileNetV2
                    model.mobilenet.classifier = nn.Sequential(
                        nn.BatchNorm1d(num_features),
                        nn.Dropout(0.3),
                        nn.Linear(num_features, 512),
                        nn.ReLU(),
                        nn.BatchNorm1d(512),
                        nn.Dropout(0.2),
                        nn.Linear(512, len(osteo_classes))
                    ).to(device)
                
                # Apply stain normalization to Osteosarcoma if using same method
                if stain_method != 'none':
                    normalized_osteo_loaders = create_stain_normalized_dataloaders(
                        osteo_dataloaders, stain_normalizer
                    )
                else:
                    normalized_osteo_loaders = osteo_dataloaders
                
                # Fine-tune on Osteosarcoma
                model = train_optimized_model(
                    model, normalized_osteo_loaders, osteo_sizes, device, 
                    num_epochs=8, apply_feature_alignment=hasattr(model, 'enable_feature_alignment')
                )
                
                # Phase 3: Final evaluation
                results = evaluate_all_splits(
                    model, normalized_osteo_loaders, osteo_sizes, device, config_name,
                    apply_feature_alignment=hasattr(model, 'enable_feature_alignment')
                )
                
                all_results[config_name] = results
                
                # Save model
                torch.save(model.state_dict(), f'./comprehensive_{config_name}_model.pth')
                
            except Exception as e:
                print(f"❌ Error in {config_name}: {e}")
                all_results[config_name] = {'error': str(e)}
    
    # Final comprehensive comparison
    print(f"\n{'='*100}")
    print("🏆 FINAL COMPREHENSIVE COMPARISON - STAIN NORM + ADVANCED TRAINING")
    print(f"{'='*100}")
    
    print(f"{'Configuration':<50} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 100)
    
    best_acc = 0
    best_config = ""
    
    for config_name, results in all_results.items():
        if 'error' not in results:
            train_acc = results['train']['accuracy']
            val_acc = results['validation']['accuracy']
            test_acc = results['test']['accuracy']
            test_f1 = results['test']['f1']
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_config = config_name
            
            print(f"{config_name:<50} {train_acc:<12.4f} {val_acc:<12.4f} {test_acc:<12.4f} {test_f1:<12.4f}")
        else:
            print(f"{config_name:<50} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
    
    print("-" * 100)
    print(f"🥇 Best Configuration: {best_config}")
    print(f"🎯 Best Test Accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    
    if best_acc >= 0.95:
        print("🎉 TARGET ACHIEVED: 95%+ accuracy!")
    else:
        print(f"📈 Progress: {best_acc*100:.1f}% (Target: 95%)")
    
    return all_results

if __name__ == "__main__":
    results = main()
