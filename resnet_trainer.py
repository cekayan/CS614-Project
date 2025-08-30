import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dataloader import create_dataloaders

class ResNetClassifier(nn.Module):
    """
    ResNet50 model with advanced classifier for 3-class classification.
    Uses transfer learning with pre-trained weights and optimized architecture.
    """
    def __init__(self, num_classes=3, freeze_features=False):
        super(ResNetClassifier, self).__init__()
        
        # Load pre-trained ResNet50 (using V2 weights like optimized trainer)
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        # Optionally freeze the feature extraction layers
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace with simple single-layer classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, num_epochs=25, device='cuda'):
    """
    Train the model and return training history.
    """
    since = time.time()
    
    # Create a copy of the model to keep track of best weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Step the scheduler after each epoch
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save to history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the model if it's the best validation accuracy
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def evaluate_model(model, dataloader, dataset_size, criterion, device='cuda', phase_name='Test'):
    """
    Evaluate the model on a given dataset and return comprehensive metrics.
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
            
            # Store predictions and labels for metrics calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / dataset_size
    acc = running_corrects.double() / dataset_size
    
    # Calculate precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Also calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f'{phase_name} Loss: {loss:.4f} Acc: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')
    
    metrics = {
        'loss': loss,
        'accuracy': acc.item(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }
    
    return metrics, all_preds, all_labels

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./resnet_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_confusion_matrix(all_preds, all_labels, class_names):
    """
    Print a simple confusion matrix.
    """
    from collections import defaultdict
    
    # Create confusion matrix
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label][pred] += 1
    
    print("\nConfusion Matrix:")
    print("Predicted ->")
    print("Actual ↓")
    print(f"{'':>12}", end="")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}", end="")
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{confusion_matrix[i][j]:>12}", end="")
        print()

def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_metrics_table(train_metrics, val_metrics, test_metrics, class_names):
    """
    Print a comprehensive metrics table for all splits.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*80)
    
    # Overall metrics table
    print("\n📊 OVERALL METRICS (Macro Average)")
    print("-" * 80)
    print(f"{'Split':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Loss':<10}")
    print("-" * 80)
    print(f"{'Train':<12} {train_metrics['accuracy']:<10.4f} {train_metrics['precision']:<12.4f} {train_metrics['recall']:<10.4f} {train_metrics['f1']:<10.4f} {train_metrics['loss']:<10.4f}")
    print(f"{'Validation':<12} {val_metrics['accuracy']:<10.4f} {val_metrics['precision']:<12.4f} {val_metrics['recall']:<10.4f} {val_metrics['f1']:<10.4f} {val_metrics['loss']:<10.4f}")
    print(f"{'Test':<12} {test_metrics['accuracy']:<10.4f} {test_metrics['precision']:<12.4f} {test_metrics['recall']:<10.4f} {test_metrics['f1']:<10.4f} {test_metrics['loss']:<10.4f}")
    
    # Per-class metrics
    print(f"\n📋 PER-CLASS METRICS")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        print(f"\n🔸 Class: {class_name}")
        print(f"{'Split':<12} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 45)
        print(f"{'Train':<12} {train_metrics['precision_per_class'][i]:<12.4f} {train_metrics['recall_per_class'][i]:<10.4f} {train_metrics['f1_per_class'][i]:<10.4f}")
        print(f"{'Validation':<12} {val_metrics['precision_per_class'][i]:<12.4f} {val_metrics['recall_per_class'][i]:<10.4f} {val_metrics['f1_per_class'][i]:<10.4f}")
        print(f"{'Test':<12} {test_metrics['precision_per_class'][i]:<12.4f} {test_metrics['recall_per_class'][i]:<10.4f} {test_metrics['f1_per_class'][i]:<10.4f}")

def get_train_metrics(model, dataloader, dataset_size, criterion, device='cuda'):
    """
    Get metrics for the training set (same as evaluate_model but cleaner interface).
    """
    metrics, preds, labels = evaluate_model(model, dataloader, dataset_size, criterion, device, 'Train')
    return metrics, preds, labels

def main():
    """
    Main training pipeline.
    """
    print("Starting ResNet50 Transfer Learning Pipeline")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    print("\nLoading data...")
    osteosarcoma_data_dir = './osteosarcoma_organized'
    BATCH_SIZE = 16
    
    try:
        dataloaders, dataset_sizes, class_names = create_dataloaders(osteosarcoma_data_dir, BATCH_SIZE)
        print(f"Successfully loaded data with {len(class_names)} classes: {class_names}")
        print(f"Dataset sizes: {dataset_sizes}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Model initialization
    print("\nInitializing ResNet50 model...")
    model = ResNetClassifier(num_classes=len(class_names), freeze_features=False)
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Using transfer learning approach - smaller learning rate for pre-trained features
    # Separate parameters into feature extractor and classifier
    feature_params = []
    classifier_params = list(model.resnet.fc.parameters())
    
    # Get all parameters except the final classifier
    for name, param in model.resnet.named_parameters():
        if not name.startswith('fc.'):
            feature_params.append(param)
    
    optimizer = optim.Adam([
        {'params': classifier_params, 'lr': 1e-3},  # Higher LR for new classifier
        {'params': feature_params, 'lr': 1e-4}     # Lower LR for pre-trained features
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training
    print("\nStarting training...")
    NUM_EPOCHS = 15  # Using fewer epochs for transfer learning
    
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    # Save the trained model
    torch.save(model.state_dict(), './resnet50_osteosarcoma.pth')
    print("Model saved as 'resnet50_osteosarcoma.pth'")
    
    # Comprehensive evaluation on all splits
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION ON ALL SPLITS")
    print("="*60)
    
    # Evaluate training set
    print("\nEvaluating training set...")
    train_metrics, train_preds, train_labels = get_train_metrics(
        model, dataloaders['train'], dataset_sizes['train'], criterion, device
    )
    
    # Evaluate validation set
    print("\nEvaluating validation set...")
    val_metrics, val_preds, val_labels = evaluate_model(
        model, dataloaders['validation'], dataset_sizes['validation'], 
        criterion, device, 'Validation'
    )
    
    # Evaluate test set
    print("\nEvaluating test set...")
    test_metrics, test_preds, test_labels = evaluate_model(
        model, dataloaders['test'], dataset_sizes['test'], 
        criterion, device, 'Test'
    )
    
    # Print confusion matrices
    print("\n" + "="*60)
    print("CONFUSION MATRICES")
    print("="*60)
    
    print("\n🔹 VALIDATION SET CONFUSION MATRIX:")
    print_confusion_matrix(val_preds, val_labels, class_names)
    
    print("\n🔹 TEST SET CONFUSION MATRIX:")
    print_confusion_matrix(test_preds, test_labels, class_names)
    
    # Plot training history
    print("\nGenerating training plots...")
    plot_training_history(history)
    
    # Print comprehensive metrics table
    print_metrics_table(train_metrics, val_metrics, test_metrics, class_names)
    
    # Model summary
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"Architecture: ResNet50")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Training Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("Training complete! 🎉")

if __name__ == '__main__':
    main()