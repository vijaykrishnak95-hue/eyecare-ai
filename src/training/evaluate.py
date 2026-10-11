from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt

from model import model, device


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

TEST_DIR = (
    PROJECT_DIR
    / "dataset"
    / "dataset2_balanced"
    / "test"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "eye_disease_efficientnet.pth"
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


print("=" * 60)
print("EYE DISEASE MODEL EVALUATION")
print("=" * 60)

print(f"Test images: {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")


# --------------------------------------------------
# Load best trained model
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


print(
    f"Best validation accuracy: "
    f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
)


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

# --------------------------------------------------
# Calculate accuracy
# --------------------------------------------------

correct = sum(
    true == pred
    for true, pred in zip(all_labels, all_predictions)
)

total = len(all_labels)

accuracy = correct / total


print("\n" + "=" * 60)
print("TEST ACCURACY")
print("=" * 60)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Calculate confusion matrix manually
# --------------------------------------------------
class_names = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]

num_classes = len(class_names)

cm = [
    [0 for _ in range(num_classes)]
    for _ in range(num_classes)
]

for true_label, predicted_label in zip(
    all_labels,
    all_predictions
):
    cm[true_label][predicted_label] += 1


# --------------------------------------------------
# Calculate precision, recall and F1-score
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report_lines = []

for class_index, class_name in enumerate(class_names):

    true_positive = cm[class_index][class_index]

    false_positive = sum(
        cm[row][class_index]
        for row in range(num_classes)
    ) - true_positive

    false_negative = sum(
        cm[class_index]
    ) - true_positive

    if true_positive + false_positive > 0:
        precision = (
            true_positive /
            (true_positive + false_positive)
        )
    else:
        precision = 0.0

    if true_positive + false_negative > 0:
        recall = (
            true_positive /
            (true_positive + false_negative)
        )
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = (
            2 * precision * recall /
            (precision + recall)
        )
    else:
        f1_score = 0.0

    line = (
        f"{class_name:25s} "
        f"Precision: {precision:.4f}  "
        f"Recall: {recall:.4f}  "
        f"F1-score: {f1_score:.4f}"
    )

    print(line)

    report_lines.append(line)


# --------------------------------------------------
# Print confusion matrix
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = True Class")
print("Columns = Predicted Class")
print()

print("Classes:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print()

for row in cm:
    print(row)


# --------------------------------------------------
# Save classification report
# --------------------------------------------------

report_path = (
    RESULTS_DIR / "classification_report.txt"
)

with open(report_path, "w") as file:

    file.write(
        "Eye Disease Classification Report\n"
    )

    file.write("=" * 50 + "\n\n")

    file.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    for line in report_lines:
        file.write(line + "\n")

    file.write("\nConfusion Matrix:\n\n")

    for row in cm:
        file.write(str(row) + "\n")


# --------------------------------------------------
# Create confusion matrix image
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title(
    "Eye Disease Classification - Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(num_classes),
    class_names,
    rotation=30
)

plt.yticks(
    range(num_classes),
    class_names
)


# Write numbers inside the matrix

for i in range(num_classes):

    for j in range(num_classes):

        plt.text(
            j,
            i,
            str(cm[i][j]),
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()


cm_path = (
    RESULTS_DIR / "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Report saved to:\n{report_path}"
)

print(
    f"\nConfusion matrix saved to:\n{cm_path}"
)