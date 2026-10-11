
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src" / "xai"))

from gradcam import generate_gradcam

# Use one image from the test dataset
image_path = (
    PROJECT_DIR
    / "dataset"
    / "dataset2_balanced"
    / "test"
    / "normal"
)

image_files = [
    p for p in image_path.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
]

if not image_files:
    raise FileNotFoundError(f"No images found in {image_path}")

result = generate_gradcam(image_files[0])

print("\nGRAD-CAM TEST RESULT")
print("Prediction:", result["prediction"])
print(f"Confidence: {result['confidence']:.2f}%")
print("Heatmap:", result["heatmap_path"])
print("Overlay:", result["overlay_path"])
