#!/usr/bin/env python3
"""
Advanced BreakHis Transfer Learning Script - Optimized for 95% Accuracy
=====================================================================

Improvements for higher accuracy:
1. More sophisticated training strategies
2. Advanced augmentation techniques
3. Ensemble methods
4. Hyperparameter optimization
5. Better architecture choices
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
    create_breakhis_dataloaders_from_raw, BreakHisDataset,
    count_trainable_parameters, debug_parameter_breakdown
)
from dataloader import create_dataloaders
from sklearn.metrics import precision_recall_fscore_support

class AdvancedBreakHisResNet(nn.Module):
    """
    Advanced ResNet50 with better architecture for transfer learning.
    """
    def __init__(self, num_classes=2):
        super(AdvancedBreakHisResNet, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')  # Use V2 weights (better)
        
        # Freeze all layers first
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze only layer4 (most important) + layer3 
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
            
        # More sophisticated classifier
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
        
        # Ensure the new classifier layers are trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.resnet(x)

class AdvancedBreakHisMobileNet(nn.Module):
    """
    Advanced MobileNetV2 with better architecture.
    """
    def __init__(self, num_classes=2):
        super(AdvancedBreakHisMobileNet, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V2')
        
        # Freeze all layers first
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
        # Unfreeze last 3 blocks for better performance
        for i in [16, 17, 18]:
            for param in self.mobilenet.features[i].parameters():
                param.requires_grad = True
        
        # Better classifier
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

def create_advanced_augmentations():
    """Create more sophisticated data augmentations."""
    # Advanced training transforms
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
    ])
    
    # Test time augmentation
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms, tta_transforms

def train_with_advanced_techniques(model, dataloaders, dataset_sizes, device, num_epochs=15):
    """
    Advanced training with multiple techniques for better performance.
    """
    # Advanced optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    # Separate learning rates for different parts
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
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-6)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early stopping
    patience = 5
    patience_counter = 0
    
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
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
            
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Early stopping and best model saving
            if phase == 'validation':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        if patience_counter >= patience:
            break
        print()
    
    print(f'Best validation accuracy: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def test_time_augmentation_predict(model, dataloader, tta_transforms, device):
    """
    Use test-time augmentation for better predictions.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_predictions = []
            
            # Apply each TTA transform
            for transform in tta_transforms:
                tta_inputs = torch.stack([transform(transforms.ToPILImage()(img)) 
                                        for img in inputs])
                tta_inputs = tta_inputs.to(device)
                
                outputs = model(tta_inputs)
                predictions = torch.softmax(outputs, dim=1)
                batch_predictions.append(predictions)
            
            # Average predictions across augmentations
            avg_predictions = torch.mean(torch.stack(batch_predictions), dim=0)
            _, preds = torch.max(avg_predictions, 1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_predictions, all_labels

def main():
    """
    Advanced training pipeline targeting 95% accuracy.
    """
    print("🚀 Advanced BreakHis Transfer Learning - Target: 95% Accuracy")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load BreakHis dataset
    print("\n📥 Loading BreakHis dataset...")
    raw_dataset_path = download_breakhis_dataset()
    if raw_dataset_path is None:
        return
    
    all_images, _, _ = create_breakhis_dataset_from_raw(raw_dataset_path)
    if all_images is None:
        return
    
    # Create dataloaders with advanced augmentations
    train_transforms, val_test_transforms, tta_transforms = create_advanced_augmentations()
    
    # Custom dataloader creation with advanced augmentations
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
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
        'validation': torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'validation': len(val_dataset), 'test': len(test_dataset)}
    
    # Load osteosarcoma dataset
    osteo_dataloaders, osteo_sizes, osteo_classes = create_dataloaders('./osteosarcoma_organized', batch_size=16)
    
    print(f"BreakHis dataset sizes: {dataset_sizes}")
    print(f"Osteosarcoma dataset sizes: {osteo_sizes}")
    
    # Train advanced models
    models_to_train = [
        ("Advanced ResNet50", AdvancedBreakHisResNet),
        ("Advanced MobileNetV2", AdvancedBreakHisMobileNet)
    ]
    
    trained_models = {}
    
    for model_name, model_class in models_to_train:
        print(f"\n🔥 Training {model_name} on BreakHis...")
        
        model = model_class(num_classes=2).to(device)
        total_params, trainable_params = count_trainable_parameters(model)
        print(f"{model_name} - Trainable: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        # Train on BreakHis
        model = train_with_advanced_techniques(model, dataloaders, dataset_sizes, device, num_epochs=20)
        
        # Transfer to osteosarcoma
        print(f"\n🔄 Transferring {model_name} to osteosarcoma...")
        
        # Freeze BreakHis-trained features
        for param in model.parameters():
            if param.requires_grad and ('layer' in str(param) or 'features' in str(param)):
                param.requires_grad = False
        
        # Replace classifier for 3 classes
        if hasattr(model, 'resnet'):
            model.resnet.fc = nn.Sequential(
                nn.BatchNorm1d(2048),
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, len(osteo_classes))
            ).to(device)
        else:
            model.mobilenet.classifier = nn.Sequential(
                nn.BatchNorm1d(1280),
                nn.Dropout(0.3),
                nn.Linear(1280, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, len(osteo_classes))
            ).to(device)
        
        # Fine-tune on osteosarcoma
        model = train_with_advanced_techniques(model, osteo_dataloaders, osteo_sizes, device, num_epochs=10)
        
        trained_models[model_name] = model
    
    # Ensemble prediction for even better results
    print("\n🎯 Final Evaluation with Ensemble...")
    
    ensemble_preds = None
    ensemble_labels = None
    
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Use TTA for better predictions
        test_preds, test_labels = test_time_augmentation_predict(
            model, osteo_dataloaders['test'], tta_transforms, device
        )
        
        if ensemble_preds is None:
            ensemble_preds = np.array(test_preds)
            ensemble_labels = np.array(test_labels)
        else:
            ensemble_preds = np.vstack([ensemble_preds, test_preds])
    
    # Majority voting ensemble
    if len(trained_models) > 1:
        final_preds = []
        for i in range(len(ensemble_labels)):
            votes = ensemble_preds[:, i]
            final_pred = np.bincount(votes).argmax()
            final_preds.append(final_pred)
        
        ensemble_acc = np.mean(np.array(final_preds) == ensemble_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ensemble_labels, final_preds, average='macro', zero_division=0
        )
        
        print(f"\n🏆 ENSEMBLE RESULTS:")
        print(f"Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        if ensemble_acc >= 0.95:
            print("🎉 TARGET ACHIEVED: 95%+ accuracy!")
        else:
            print(f"📈 Progress: {ensemble_acc*100:.1f}% (Target: 95%)")

if __name__ == '__main__':
    main()