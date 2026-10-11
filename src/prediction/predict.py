import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "eye_disease_efficientnet.pth"

# Class names must match the training order
CLASS_NAMES = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the trained EfficientNet-B0 model."""
    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_path, model=None):
    """Return the predicted class and confidence percentage."""
    if model is None:
        model = load_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    return {
        "disease": CLASS_NAMES[predicted_index.item()],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(probabilities[0, i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }