from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# Dataset location
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "dataset2_balanced"

# Classes
CLASSES = [
    "normal",
    "cataract",
    "diabetic_retinopathy",
    "glaucoma"
]

# Number of samples per class
SAMPLES_PER_CLASS = 3

# Create one figure
fig, axes = plt.subplots(
    len(CLASSES),
    SAMPLES_PER_CLASS,
    figsize=(12, 12)
)

for row, class_name in enumerate(CLASSES):

    class_dir = DATASET_DIR / "train" / class_name

    image_files = [
        file for file in class_dir.iterdir()
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    selected_images = random.sample(
        image_files,
        min(SAMPLES_PER_CLASS, len(image_files))
    )

    for col, image_path in enumerate(selected_images):

        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.suptitle(
    "Eye Disease Dataset - Sample Training Images",
    fontsize=16
)

plt.tight_layout()

# Save the result
output_path = Path(__file__).resolve().parents[2] / "results" / "sample_images.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Sample image grid saved to: {output_path}")

plt.show()