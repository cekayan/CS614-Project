import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import ImageFile

# --- FIX for Truncated Images ---
# This line tells the PIL library to be tolerant of corrupted image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_dataloaders(data_dir, batch_size=16):
    """
    Creates PyTorch data loaders for training, validation, and testing.

    Args:
        data_dir (str): The root directory of the dataset 
                        (e.g., '.../datasets/osteosarcoma_organized/').
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders,
               and a dictionary of dataset sizes.
    """
    
    # Define image size as specified in the paper
    IMAGE_SIZE = 224

    # --- THIS IS YOUR PREPROCESSING AND AUGMENTATION PIPELINE ---
    
    # Transformations for the training set (with data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(), # Converts image to a PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizes tensor
    ])

    # Transformations for the validation and test sets (only resizing and normalization)
    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- Create Datasets using ImageFolder ---
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
        'validation': datasets.ImageFolder(os.path.join(data_dir, 'validation'), val_test_transforms),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), val_test_transforms)
    }

    # --- Create DataLoaders ---
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    return dataloaders, dataset_sizes, class_names

if __name__ == '__main__':
    # --- How to use the function ---
    
    # 1. Define the path to your organized dataset
    # This should be the output of your reorganizer script
    osteosarcoma_data_dir = './osteosarcoma_organized' 

    # 2. Define your batch size
    BATCH_SIZE = 16

    # 3. Create the dataloaders
    if os.path.isdir(osteosarcoma_data_dir):
        dataloaders, dataset_sizes, class_names = create_dataloaders(osteosarcoma_data_dir, BATCH_SIZE)

        # 4. Example: Get one batch of training images and labels
        try:
            inputs, classes = next(iter(dataloaders['train']))
            print(f"\nSuccessfully loaded one batch.")
            print(f"Batch of inputs shape: {inputs.shape}") 
            print(f"Batch of labels shape: {classes.shape}") 
        except StopIteration:
            print("The train dataloader is empty. Check your 'train' folder.")
        except Exception as e:
            print(f"An error occurred while loading a batch: {e}")

    else:
        print(f"Error: The directory '{osteosarcoma_data_dir}' does not exist.")

