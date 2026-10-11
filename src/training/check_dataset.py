from pathlib import Path

# Project dataset location
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset" / "dataset2_balanced"

# Expected classes
CLASSES = [
    "normal",
    "cataract",
    "diabetic_retinopathy",
    "glaucoma"
]

# Dataset splits
SPLITS = ["train", "val", "test"]

# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder):
    """Count image files inside a folder."""
    return sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


print("=" * 60)
print("EYE DISEASE DATASET VERIFICATION")
print("=" * 60)

print(f"\nDataset location:\n{DATASET_DIR}")

if not DATASET_DIR.exists():
    print("\nERROR: Dataset folder was not found.")
    print("Check that dataset2_balanced is inside the dataset folder.")
    raise SystemExit

total_images = 0

for split in SPLITS:
    split_path = DATASET_DIR / split

    print(f"\n--- {split.upper()} ---")

    if not split_path.exists():
        print(f"ERROR: Missing {split} folder.")
        continue

    split_total = 0

    for class_name in CLASSES:
        class_path = split_path / class_name

        if not class_path.exists():
            print(f"{class_name}: MISSING")
            continue

        count = count_images(class_path)
        split_total += count

        print(f"{class_name}: {count}")

    print(f"Total {split}: {split_total}")

    total_images += split_total

print("\n" + "=" * 60)
print(f"TOTAL DATASET IMAGES: {total_images}")
print("=" * 60)