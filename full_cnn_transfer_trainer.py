#!/usr/bin/env python3
"""
Full CNN Transfer Learning Script
=================================

This script:
1. Trains the ENTIRE CNN (ResNet50/MobileNetV2/DenseNet121) on BreakHis with simple classifier
2. Removes the classifier and uses pre-trained CNN features for osteosarcoma classification
3. Evaluates feature extraction capability of fully-trained CNNs

The approach:
- Load ImageNet pre-trained models
- Train ALL layers (full CNN + simple classifier) on BreakHis
- Remove classifier, freeze CNN, add new classifier for osteosarcoma
- Compare feature extraction quality across architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import time
import copy
import numpy as np
from breakhis_transfer_trainer import (
    download_breakhis_dataset, create_breakhis_dataset_from_raw, 
    BreakHisDataset, create_breakhis_dataloaders_from_raw
)
from dataloader import create_dataloaders
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from stain_normalization import ReinhardNormalizer

class StainNormalizedDataset(Dataset):
    """Dataset wrapper that applies Reinhard stain normalization on-the-fly."""
    
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

def fit_reinhard_normalizer(source_dataset, target_dataset):
    """Fit Reinhard normalizer using target domain characteristics."""
    print("  Fitting Reinhard stain normalizer...")
    
    # Get a representative target image (Osteosarcoma)
    target_image, _ = target_dataset[0]
    
    # Create and fit normalizer
    normalizer = ReinhardNormalizer()
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

def compute_domain_statistics(dataloader, model, device):
    """Compute feature statistics for domain alignment."""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = model.get_features(inputs)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    
    # Compute statistics
    mean = torch.mean(all_features, dim=0)
    std = torch.std(all_features, dim=0)
    
    return mean, std

def apply_feature_normalization(features, source_stats, target_stats):
    """Apply statistical moment matching for feature alignment."""
    source_mean, source_std = source_stats
    target_mean, target_std = target_stats
    
    # Normalize to zero mean, unit variance
    normalized_features = (features - source_mean.to(features.device)) / (source_std.to(features.device) + 1e-8)
    
    # Scale to target distribution
    aligned_features = normalized_features * target_std.to(features.device) + target_mean.to(features.device)
    
    return aligned_features

class FullTrainingResNet(nn.Module):
    """
    ResNet50 with simple single-layer classifier for full CNN training on BreakHis.
    All layers are trainable.
    """
    def __init__(self, num_classes=2):
        super(FullTrainingResNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        # All layers are trainable (no freezing)
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        # Simple single-layer classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)
    
    def get_features(self, x):
        """Extract features without classifier."""
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

class FullTrainingMobileNet(nn.Module):
    """
    MobileNetV2 with simple single-layer classifier for full CNN training on BreakHis.
    All layers are trainable.
    """
    def __init__(self, num_classes=2):
        super(FullTrainingMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V2')
        
        # All layers are trainable (no freezing)
        for param in self.mobilenet.parameters():
            param.requires_grad = True
        
        # Simple single-layer classifier
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.mobilenet(x)
    
    def get_features(self, x):
        """Extract features without classifier."""
        x = self.mobilenet.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

class FullTrainingDenseNet(nn.Module):
    """
    DenseNet121 with simple single-layer classifier for full CNN training on BreakHis.
    All layers are trainable.
    """
    def __init__(self, num_classes=2):
        super(FullTrainingDenseNet, self).__init__()
        
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(weights='IMAGENET1K_V1')
        
        # All layers are trainable (no freezing)
        for param in self.densenet.parameters():
            param.requires_grad = True
        
        # Simple single-layer classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)
    
    def get_features(self, x):
        """Extract features without classifier."""
        features = self.densenet.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_full_cnn(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, 
                   num_epochs=15, device='cuda', model_name="Model"):
    """
    Train the entire CNN on BreakHis dataset.
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
    
    print(f"🔥 Training full {model_name} CNN on BreakHis...")
    
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
            
            print(f'{phase.capitalize()}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}')
            
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

class FeatureExtractorClassifier(nn.Module):
    """
    Wrapper that uses pre-trained CNN features with domain alignment for osteosarcoma.
    """
    def __init__(self, feature_extractor, num_features, num_classes=3, alignment_method='batch_norm'):
        super(FeatureExtractorClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.alignment_method = alignment_method
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Domain alignment layer
        if alignment_method == 'batch_norm':
            self.domain_adapter = nn.BatchNorm1d(num_features, affine=True)
        elif alignment_method == 'layer_norm':
            self.domain_adapter = nn.LayerNorm(num_features)
        elif alignment_method == 'instance_norm':
            self.domain_adapter = nn.InstanceNorm1d(num_features, affine=True)
        else:
            self.domain_adapter = nn.Identity()  # No alignment
        
        # New classifier for osteosarcoma (3 classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Statistics for moment matching
        if alignment_method == 'moment_matching':
            self.register_buffer('source_mean', torch.zeros(num_features))
            self.register_buffer('source_std', torch.ones(num_features))
            self.register_buffer('target_mean', torch.zeros(num_features))
            self.register_buffer('target_std', torch.ones(num_features))
    
    def set_domain_statistics(self, source_stats, target_stats):
        """Set statistics for moment matching."""
        if self.alignment_method == 'moment_matching':
            source_mean, source_std = source_stats
            target_mean, target_std = target_stats
            
            self.source_mean.copy_(source_mean)
            self.source_std.copy_(source_std)
            self.target_mean.copy_(target_mean)
            self.target_std.copy_(target_std)
        
    def forward(self, x):
        # Extract features using pre-trained CNN
        features = self.feature_extractor.get_features(x)
        
        # Apply domain alignment
        if self.alignment_method == 'moment_matching':
            aligned_features = apply_feature_normalization(
                features, (self.source_mean, self.source_std),
                (self.target_mean, self.target_std)
            )
        else:
            aligned_features = self.domain_adapter(features)
        
        # Classify using new classifier
        return self.classifier(aligned_features)

def evaluate_model_comprehensive(model, dataloader, dataset_size, criterion, device='cuda', phase_name='Test'):
    """
    Comprehensive model evaluation with all metrics.
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / dataset_size
    acc = running_corrects.double() / dataset_size
    
    # Calculate comprehensive metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print(f'{phase_name}: Loss {loss:.4f}, Acc {acc:.4f}, Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}')
    
    return {
        'loss': loss,
        'accuracy': acc.item(),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def print_comprehensive_results(results_dict):
    """
    Print comprehensive comparison table for all models and splits.
    """
    print("\n" + "="*100)
    print("🏆 FULL CNN TRANSFER LEARNING RESULTS")
    print("="*100)
    
    models = list(results_dict.keys())
    splits = ['train', 'validation', 'test']
    
    # Print results for each model
    for model_name in models:
        print(f"\n📊 {model_name} (BreakHis Full Training → Osteosarcoma Feature Extraction)")
        print("-" * 80)
        print(f"{'Split':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Loss':<10}")
        print("-" * 80)
        
        for split in splits:
            r = results_dict[model_name][split]
            print(f"{split.capitalize():<12} {r['accuracy']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['loss']:<10.4f}")
    
    # Comparison table (Test set only)
    print(f"\n🥇 TEST SET COMPARISON")
    print("-" * 80)
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    test_results = []
    for model_name in models:
        r = results_dict[model_name]['test']
        test_results.append((model_name, r['accuracy']))
        print(f"{model_name:<15} {r['accuracy']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}")
    
    # Find best model
    best_model, best_acc = max(test_results, key=lambda x: x[1])
    print("-" * 80)
    print(f"🏆 Best Model: {best_model} (Test Accuracy: {best_acc:.4f})")

def main():
    """
    Main pipeline for full CNN training and feature extraction evaluation.
    """
    print("🔬 Full CNN Training → Feature Extraction Transfer Learning")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Download and prepare BreakHis dataset
    print("\n" + "="*60)
    print("STEP 1: DATASET PREPARATION")
    print("="*60)
    
    print("📥 Downloading BreakHis dataset...")
    raw_dataset_path = download_breakhis_dataset()
    
    if raw_dataset_path is None:
        print("❌ Failed to download BreakHis dataset")
        return
    
    # Create dataset from raw structure
    all_images, _, _ = create_breakhis_dataset_from_raw(raw_dataset_path)
    if all_images is None:
        print("❌ Failed to parse BreakHis dataset structure")
        return
    
    # Create BreakHis dataloaders
    breakhis_dataloaders, breakhis_sizes, breakhis_classes = create_breakhis_dataloaders_from_raw(all_images, batch_size=32)
    
    # Load osteosarcoma dataset
    print("\nLoading osteosarcoma dataset...")
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=32)
    
    print(f"BreakHis: {breakhis_classes} (samples: {breakhis_sizes})")
    print(f"Osteosarcoma: {osteo_classes} (samples: {osteo_sizes})")
    
    # Models to train
    models_config = [
        ("ResNet50", FullTrainingResNet, 2048),
        ("MobileNetV2", FullTrainingMobileNet, 1280),
        ("DenseNet121", FullTrainingDenseNet, 1024)
    ]
    
    # Stain normalization experiments: both with and without Reinhard
    stain_experiments = [
        ("None", None),
        ("Reinhard", "reinhard")
    ]
    
    trained_models = {}
    results = {}
    
    # Step 2: Train each CNN fully on BreakHis (with and without stain normalization)
    print("\n" + "="*60)
    print("STEP 2: FULL CNN TRAINING ON BREAKHIS")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    
    for stain_name, stain_method in stain_experiments:
        print(f"\n🎨 STAIN NORMALIZATION: {stain_name.upper()}")
        print("=" * 60)
        
        # Prepare stain normalizer if needed
        if stain_method == "reinhard":
            print("Setting up Reinhard stain normalization...")
            stain_normalizer = fit_reinhard_normalizer(
                breakhis_dataloaders['train'].dataset, 
                osteo_dataloaders['train'].dataset
            )
            # Apply stain normalization to BreakHis data
            current_breakhis_dataloaders = create_stain_normalized_dataloaders(
                breakhis_dataloaders, stain_normalizer
            )
        else:
            current_breakhis_dataloaders = breakhis_dataloaders
            stain_normalizer = None
        
        for model_name, model_class, num_features in models_config:
            experiment_name = f"{stain_name}_{model_name}"
            print(f"\n🚀 Training {experiment_name}...")
            print("-" * 50)
            
            # Initialize model
            model = model_class(num_classes=len(breakhis_classes))
            model = model.to(device)
            
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,} (100%)")
            
            # Optimizer and scheduler for full training
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            
            # Train the full CNN
            trained_model, history = train_full_cnn(
                model, current_breakhis_dataloaders, breakhis_sizes,
                criterion, optimizer, scheduler, num_epochs=15, device=device, model_name=experiment_name
            )
            
            trained_models[experiment_name] = (trained_model, num_features, stain_normalizer)
            
            # Save the fully trained model
            torch.save(trained_model.state_dict(), f'./full_trained_{experiment_name.lower()}_breakhis.pth')
            print(f"✅ {experiment_name} saved as 'full_trained_{experiment_name.lower()}_breakhis.pth'")
    
    # Step 3: Feature extraction evaluation on osteosarcoma
    print("\n" + "="*60)
    print("STEP 3: FEATURE EXTRACTION ON OSTEOSARCOMA")
    print("="*60)
    
    for experiment_name, (trained_model, num_features, stain_normalizer) in trained_models.items():
        print(f"\n🔍 Evaluating {experiment_name} feature extraction...")
        print("-" * 50)
        
        # Prepare osteosarcoma dataloaders with same stain normalization if used
        if stain_normalizer is not None:
            print("Applying same stain normalization to osteosarcoma data...")
            current_osteo_dataloaders = create_stain_normalized_dataloaders(
                osteo_dataloaders, stain_normalizer
            )
        else:
            current_osteo_dataloaders = osteo_dataloaders
        
        # Create feature extractor + new classifier with domain alignment
        feature_classifier = FeatureExtractorClassifier(
            trained_model, num_features, len(osteo_classes), 
            alignment_method='moment_matching'  # Options: 'moment_matching', 'batch_norm', 'layer_norm', 'instance_norm', 'none'
        ).to(device)
        
        # Compute dataset-aware statistics for moment matching
        if hasattr(feature_classifier, 'set_domain_statistics'):
            print("Computing domain statistics for dataset alignment...")
            # Use the appropriate dataloaders for statistics computation
            if stain_normalizer is None:
                source_dataloader = breakhis_dataloaders['train']
            else:
                # Need to recreate the normalized dataloader for this experiment
                temp_breakhis_dataloaders = create_stain_normalized_dataloaders(
                    breakhis_dataloaders, stain_normalizer
                )
                source_dataloader = temp_breakhis_dataloaders['train']
            source_stats = compute_domain_statistics(source_dataloader, trained_model, device)
            target_stats = compute_domain_statistics(current_osteo_dataloaders['train'], trained_model, device)
            feature_classifier.set_domain_statistics(source_stats, target_stats)
            print(f"Source domain - Mean: {source_stats[0][:5].numpy()}, Std: {source_stats[1][:5].numpy()}")
            print(f"Target domain - Mean: {target_stats[0][:5].numpy()}, Std: {target_stats[1][:5].numpy()}")
        
        # Train only the new classifier
        classifier_optimizer = optim.Adam(feature_classifier.classifier.parameters(), lr=1e-3)
        classifier_scheduler = optim.lr_scheduler.StepLR(classifier_optimizer, step_size=3, gamma=0.5)
        
        # Train classifier
        print("Training new classifier on osteosarcoma...")
        feature_classifier, _ = train_full_cnn(
            feature_classifier, current_osteo_dataloaders, osteo_sizes,
            criterion, classifier_optimizer, classifier_scheduler, 
            num_epochs=20, device=device, model_name=f"{experiment_name} Classifier"
        )
        
        # Evaluate on all splits
        print(f"\nEvaluating {experiment_name} on all osteosarcoma splits...")
        results[experiment_name] = {}
        
        for split in ['train', 'validation', 'test']:
            results[experiment_name][split] = evaluate_model_comprehensive(
                feature_classifier, current_osteo_dataloaders[split], osteo_sizes[split],
                criterion, device, f"{experiment_name} {split.capitalize()}"
            )
        
        # Save the feature extraction model
        torch.save(feature_classifier.state_dict(), f'./feature_extractor_{experiment_name.lower()}_osteosarcoma.pth')
    
    # Step 4: Print comprehensive results
    print("\n" + "="*60)
    print("STEP 4: COMPREHENSIVE RESULTS")
    print("="*60)
    
    print_comprehensive_results(results)
    
    print("\n✅ Full CNN transfer learning experiment completed!")
    print("\n📁 Saved Models:")
    for experiment_name, _ in trained_models.items():
        print(f"- full_trained_{experiment_name.lower()}_breakhis.pth (BreakHis trained)")
        print(f"- feature_extractor_{experiment_name.lower()}_osteosarcoma.pth (Feature extractor)")
    
    print("\n🎯 EXPERIMENT SUMMARY:")
    print("- None_*: Baseline without stain normalization")
    print("- Reinhard_*: With Reinhard stain normalization")
    print("- All models: ResNet50, MobileNetV2, DenseNet121")
    print("- Total experiments: 6 (3 models × 2 stain conditions)")

if __name__ == '__main__':
    main()