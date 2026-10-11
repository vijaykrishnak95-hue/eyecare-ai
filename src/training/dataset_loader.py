from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# 1. Dataset location
# --------------------------------------------------

DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "dataset2_balanced"
)


# --------------------------------------------------
# 2. Image settings
# --------------------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 32


# --------------------------------------------------
# 3. Image preprocessing and augmentation
# --------------------------------------------------

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 4. Load datasets
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=val_test_transforms
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=val_test_transforms
)


# --------------------------------------------------
# 5. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 6. Print dataset information
# --------------------------------------------------

print("=" * 60)
print("DATASET LOADER VERIFICATION")
print("=" * 60)

print(f"Dataset location: {DATASET_DIR}")

print(f"\nClasses: {train_dataset.classes}")
print(f"Class mapping: {train_dataset.class_to_idx}")

print(f"\nTraining images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")

print(f"\nImage size: {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

print("\nDataLoaders created successfully.")
print("=" * 60)