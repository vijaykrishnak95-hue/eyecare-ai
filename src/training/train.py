from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset_loader import train_loader, val_loader
from model import model, device


# --------------------------------------------------
# Training settings
# --------------------------------------------------

NUM_EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "eye_disease_efficientnet.pth"


# --------------------------------------------------
# Freeze EfficientNet feature extractor
# --------------------------------------------------

for parameter in model.features.parameters():
    parameter.requires_grad = False


# --------------------------------------------------
# Loss function and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_accuracy = 0.0

print("=" * 60)
print("EYE DISEASE MODEL TRAINING")
print("=" * 60)

print(f"Device: {device}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

for epoch in range(NUM_EPOCHS):

    # ----------------------------------------------
    # Training phase
    # ----------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / train_total
    train_accuracy = train_correct / train_total


    # ----------------------------------------------
    # Validation phase
    # ----------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total


    # ----------------------------------------------
    # Print epoch results
    # ----------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy * 100:.2f}%"
    )


    # ----------------------------------------------
    # Save best model
    # ----------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": train_loader.dataset.classes,
                "image_size": 224,
                "best_val_accuracy": best_val_accuracy
            },
            BEST_MODEL_PATH
        )

        print("✓ Best model saved.")


# --------------------------------------------------
# Training complete
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(f"Model saved to:\n{BEST_MODEL_PATH}")