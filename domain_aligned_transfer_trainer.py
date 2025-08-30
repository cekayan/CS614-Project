#!/usr/bin/env python3
"""
Domain-Aligned Transfer Learning Script
======================================

This script adds domain alignment techniques for better transfer from BreakHis to Osteosarcoma:
1. Feature-level normalization (BatchNorm adaptation)
2. Statistical moment matching 
3. Histogram matching for pixel-level alignment
4. Adaptive instance normalization
5. Domain adversarial training (optional)

Methods for cross-domain alignment:
- Pixel-level: Histogram matching, color normalization
- Feature-level: Batch normalization adaptation, moment matching
- Distribution-level: Maximum Mean Discrepancy (MMD)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.nn.functional as F
import time
import copy
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import cv2
from breakhis_transfer_trainer import (
    download_breakhis_dataset, create_breakhis_dataset_from_raw, 
    BreakHisDataset, create_breakhis_dataloaders_from_raw
)
from dataloader import create_dataloaders
from sklearn.metrics import precision_recall_fscore_support

class DomainAlignedResNet(nn.Module):
    """ResNet50 with domain alignment capabilities."""
    def __init__(self, num_classes=2, enable_domain_alignment=True):
        super(DomainAlignedResNet, self).__init__()
        
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.enable_domain_alignment = enable_domain_alignment
        
        # All layers trainable
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        # Simple classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Domain alignment layers
        if enable_domain_alignment:
            self.feature_adapter = nn.Sequential(
                nn.BatchNorm1d(num_features, affine=True),
                nn.Dropout(0.1)
            )
        
    def forward(self, x, adapt_domain=False):
        # Extract features
        features = self.get_features(x)
        
        # Apply domain adaptation if enabled
        if adapt_domain and self.enable_domain_alignment:
            features = self.feature_adapter(features)
        
        # Classification
        return self.resnet.fc(features)
    
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

class DomainAlignedMobileNet(nn.Module):
    """MobileNetV2 with domain alignment capabilities."""
    def __init__(self, num_classes=2, enable_domain_alignment=True):
        super(DomainAlignedMobileNet, self).__init__()
        
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V2')
        self.enable_domain_alignment = enable_domain_alignment
        
        # All layers trainable
        for param in self.mobilenet.parameters():
            param.requires_grad = True
        
        # Simple classifier
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Linear(num_features, num_classes)
        
        # Domain alignment layers
        if enable_domain_alignment:
            self.feature_adapter = nn.Sequential(
                nn.BatchNorm1d(num_features, affine=True),
                nn.Dropout(0.1)
            )
        
    def forward(self, x, adapt_domain=False):
        features = self.get_features(x)
        
        if adapt_domain and self.enable_domain_alignment:
            features = self.feature_adapter(features)
        
        return self.mobilenet.classifier(features)
    
    def get_features(self, x):
        """Extract features without classifier."""
        x = self.mobilenet.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

class DomainAlignedDenseNet(nn.Module):
    """DenseNet121 with domain alignment capabilities."""
    def __init__(self, num_classes=2, enable_domain_alignment=True):
        super(DomainAlignedDenseNet, self).__init__()
        
        self.densenet = models.densenet121(weights='IMAGENET1K_V1')
        self.enable_domain_alignment = enable_domain_alignment
        
        # All layers trainable
        for param in self.densenet.parameters():
            param.requires_grad = True
        
        # Simple classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes)
        
        # Domain alignment layers
        if enable_domain_alignment:
            self.feature_adapter = nn.Sequential(
                nn.BatchNorm1d(num_features, affine=True),
                nn.Dropout(0.1)
            )
        
    def forward(self, x, adapt_domain=False):
        features = self.get_features(x)
        
        if adapt_domain and self.enable_domain_alignment:
            features = self.feature_adapter(features)
        
        return self.densenet.classifier(features)
    
    def get_features(self, x):
        """Extract features without classifier."""
        features = self.densenet.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

def histogram_matching(source_img, target_img):
    """
    Apply histogram matching to align source image to target distribution.
    """
    # Convert to numpy if tensor
    if torch.is_tensor(source_img):
        source_img = source_img.numpy()
    if torch.is_tensor(target_img):
        target_img = target_img.numpy()
    
    # Apply histogram matching per channel
    matched_img = np.zeros_like(source_img)
    for channel in range(source_img.shape[0]):
        matched_img[channel] = match_histograms(
            source_img[channel], target_img[channel]
        )
    
    return torch.from_numpy(matched_img).float()

def match_histograms(source, template):
    """Match histogram of source to template using cumulative distribution functions."""
    # Get unique values and their counts
    s_values, s_counts = np.unique(source.flatten(), return_counts=True)
    t_values, t_counts = np.unique(template.flatten(), return_counts=True)
    
    # Calculate CDFs
    s_cdf = np.cumsum(s_counts).astype(float) / source.size
    t_cdf = np.cumsum(t_counts).astype(float) / template.size
    
    # Create mapping function
    interp_t_values = np.interp(s_cdf, t_cdf, t_values)
    
    # Map source values to template distribution
    matched = np.interp(source.flatten(), s_values, interp_t_values)
    
    return matched.reshape(source.shape)

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

class DomainAlignedFeatureExtractor(nn.Module):
    """Feature extractor with domain alignment for osteosarcoma classification."""
    def __init__(self, feature_extractor, num_features, num_classes=3, 
                 alignment_method='batch_norm'):
        super(DomainAlignedFeatureExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        self.alignment_method = alignment_method
        
        # Freeze feature extractor
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
            self.domain_adapter = nn.Identity()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        # Statistics for moment matching
        self.register_buffer('source_mean', torch.zeros(num_features))
        self.register_buffer('source_std', torch.ones(num_features))
        self.register_buffer('target_mean', torch.zeros(num_features))
        self.register_buffer('target_std', torch.ones(num_features))
        
    def set_domain_statistics(self, source_stats, target_stats):
        """Set statistics for moment matching."""
        source_mean, source_std = source_stats
        target_mean, target_std = target_stats
        
        self.source_mean.copy_(source_mean)
        self.source_std.copy_(source_std)
        self.target_mean.copy_(target_mean)
        self.target_std.copy_(target_std)
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor.get_features(x)
        
        # Apply domain alignment
        if self.alignment_method == 'moment_matching':
            features = apply_feature_normalization(
                features, (self.source_mean, self.source_std),
                (self.target_mean, self.target_std)
            )
        else:
            features = self.domain_adapter(features)
        
        # Classification
        return self.classifier(features)

def create_domain_aligned_transforms(reference_dataset=None):
    """Create transforms with domain alignment preprocessing."""
    
    # Standard transforms
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    
    # Add domain-specific normalization
    if reference_dataset == 'breakhis':
        # BreakHis-optimized normalization
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Standard ImageNet
            std=[0.229, 0.224, 0.225]
        )
    elif reference_dataset == 'osteosarcoma':
        # Osteosarcoma-optimized normalization (adjust based on dataset statistics)
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Can be computed from dataset
            std=[0.229, 0.224, 0.225]
        )
    else:
        # Standard ImageNet normalization
        normalization = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    return transforms.Compose(base_transforms + [normalization])

def train_with_domain_alignment(model, dataloaders, dataset_sizes, criterion, optimizer, 
                               scheduler=None, num_epochs=15, device='cuda', model_name="Model",
                               alignment_method='batch_norm'):
    """Training with domain alignment techniques."""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print(f"🔥 Training {model_name} with {alignment_method} domain alignment...")
    
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
                    # Use domain adaptation during training
                    if hasattr(model, 'forward') and 'adapt_domain' in model.forward.__code__.co_varnames:
                        outputs = model(inputs, adapt_domain=True)
                    else:
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
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model, {}

def main():
    """Main pipeline with domain alignment techniques."""
    print("🔬 Domain-Aligned Transfer Learning Experiment")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Domain alignment methods to test
    alignment_methods = [
        'batch_norm',      # Batch normalization adaptation
        'layer_norm',      # Layer normalization
        'instance_norm',   # Instance normalization  
        'moment_matching', # Statistical moment matching
        'none'            # No alignment (baseline)
    ]
    
    print(f"Testing alignment methods: {alignment_methods}")
    
    # Step 1: Dataset preparation
    print("\n" + "="*60)
    print("STEP 1: DATASET PREPARATION")
    print("="*60)
    
    # Download BreakHis
    raw_dataset_path = download_breakhis_dataset()
    if raw_dataset_path is None:
        return
    
    all_images, _, _ = create_breakhis_dataset_from_raw(raw_dataset_path)
    if all_images is None:
        return
    
    breakhis_dataloaders, breakhis_sizes, breakhis_classes = create_breakhis_dataloaders_from_raw(all_images, batch_size=32)
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=32)
    
    print(f"BreakHis: {breakhis_classes} (samples: {breakhis_sizes})")
    print(f"Osteosarcoma: {osteo_classes} (samples: {osteo_sizes})")
    
    # Models to test
    models_config = [
        ("ResNet50", DomainAlignedResNet, 2048),
        ("MobileNetV2", DomainAlignedMobileNet, 1280), 
        ("DenseNet121", DomainAlignedDenseNet, 1024)
    ]
    
    # Step 2: Train base models on BreakHis
    print("\n" + "="*60)
    print("STEP 2: TRAINING BASE MODELS ON BREAKHIS")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    trained_models = {}
    
    for model_name, model_class, num_features in models_config:
        print(f"\n🚀 Training {model_name}...")
        
        model = model_class(num_classes=len(breakhis_classes), enable_domain_alignment=True)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        trained_model, _ = train_with_domain_alignment(
            model, breakhis_dataloaders, breakhis_sizes,
            criterion, optimizer, scheduler, num_epochs=15, 
            device=device, model_name=model_name
        )
        
        trained_models[model_name] = (trained_model, num_features)
    
    # Step 3: Domain alignment evaluation
    print("\n" + "="*60)
    print("STEP 3: DOMAIN ALIGNMENT EVALUATION")
    print("="*60)
    
    results = {}
    
    for alignment_method in alignment_methods:
        print(f"\n🔍 Testing {alignment_method} alignment...")
        results[alignment_method] = {}
        
        for model_name, (trained_model, num_features) in trained_models.items():
            print(f"\n--- {model_name} with {alignment_method} ---")
            
            # Create domain-aligned feature extractor
            feature_extractor = DomainAlignedFeatureExtractor(
                trained_model, num_features, len(osteo_classes), alignment_method
            ).to(device)
            
            # Compute domain statistics for moment matching
            if alignment_method == 'moment_matching':
                print("Computing domain statistics...")
                source_stats = compute_domain_statistics(breakhis_dataloaders['train'], trained_model, device)
                target_stats = compute_domain_statistics(osteo_dataloaders['train'], trained_model, device)
                feature_extractor.set_domain_statistics(source_stats, target_stats)
            
            # Train classifier with domain alignment
            classifier_optimizer = optim.Adam(feature_extractor.classifier.parameters(), lr=1e-3)
            classifier_scheduler = optim.lr_scheduler.StepLR(classifier_optimizer, step_size=3, gamma=0.5)
            
            feature_extractor, _ = train_with_domain_alignment(
                feature_extractor, osteo_dataloaders, osteo_sizes,
                criterion, classifier_optimizer, classifier_scheduler,
                num_epochs=10, device=device, 
                model_name=f"{model_name}-{alignment_method}",
                alignment_method=alignment_method
            )
            
            # Evaluate
            results[alignment_method][model_name] = {}
            for split in ['train', 'validation', 'test']:
                metrics = evaluate_model_comprehensive(
                    feature_extractor, osteo_dataloaders[split], osteo_sizes[split],
                    criterion, device, f"{model_name}-{alignment_method} {split}"
                )
                results[alignment_method][model_name][split] = metrics
    
    # Step 4: Print comprehensive results
    print("\n" + "="*80)
    print("🏆 DOMAIN ALIGNMENT COMPARISON RESULTS")
    print("="*80)
    
    print_domain_alignment_results(results)
    
    print("\n✅ Domain-aligned transfer learning completed!")

def evaluate_model_comprehensive(model, dataloader, dataset_size, criterion, device='cuda', phase_name='Test'):
    """Comprehensive model evaluation."""
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

def print_domain_alignment_results(results):
    """Print comprehensive domain alignment comparison."""
    
    # Test set comparison across alignment methods
    print(f"\n🥇 TEST SET ACCURACY COMPARISON")
    print("-" * 100)
    print(f"{'Alignment Method':<20} {'ResNet50':<12} {'MobileNetV2':<12} {'DenseNet121':<12} {'Average':<12}")
    print("-" * 100)
    
    method_averages = []
    
    for method in results.keys():
        accuracies = []
        row = f"{method:<20}"
        
        for model in ['ResNet50', 'MobileNetV2', 'DenseNet121']:
            if model in results[method]:
                acc = results[method][model]['test']['accuracy']
                accuracies.append(acc)
                row += f"{acc:<12.4f}"
            else:
                row += f"{'N/A':<12}"
        
        avg_acc = np.mean(accuracies) if accuracies else 0
        method_averages.append((method, avg_acc))
        row += f"{avg_acc:<12.4f}"
        print(row)
    
    # Find best alignment method
    best_method, best_avg = max(method_averages, key=lambda x: x[1])
    print("-" * 100)
    print(f"🏆 Best Alignment Method: {best_method} (Average Accuracy: {best_avg:.4f})")
    
    # Improvement over baseline
    baseline_avg = next((avg for method, avg in method_averages if method == 'none'), 0)
    if baseline_avg > 0:
        improvement = best_avg - baseline_avg
        print(f"📈 Improvement over baseline: {improvement:.4f} ({improvement/baseline_avg*100:.1f}%)")

if __name__ == '__main__':
    main()