
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "eye_disease_efficientnet.pth"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
]

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    """Load the trained EfficientNet-B0 checkpoint."""
    import sys

    training_dir = PROJECT_DIR / "src" / "training"
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))

    from model import model as base_model

    loaded_model = base_model

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    loaded_model.load_state_dict(state_dict)
    loaded_model.to(DEVICE)
    loaded_model.eval()

    return loaded_model


def generate_gradcam(image_path, model=None):
    """Generate prediction, Grad-CAM heatmap and overlay for an image."""
    image_path = Path(image_path)

    original = Image.open(image_path).convert("RGB")
    input_tensor = transform(original).unsqueeze(0).to(DEVICE)

    if model is None:
        model = load_model()

    # Locate the final convolutional layer in the model.
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise RuntimeError("No convolutional layer was found in the model.")

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_index = probabilities.max(dim=1)

            score = logits[0, predicted_index.item()]
            score.backward()

        feature_maps = activations[0]
        grads = gradients[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        resized_original = cv2.resize(
            np.array(original), (IMAGE_SIZE, IMAGE_SIZE)
        )
        overlay = cv2.addWeighted(
            resized_original, 0.6, heatmap, 0.4, 0
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        heatmap_path = RESULTS_DIR / "gradcam_heatmap.png"
        overlay_path = RESULTS_DIR / "gradcam_overlay.png"

        cv2.imwrite(str(heatmap_path), heatmap)
        cv2.imwrite(
            str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        return {
            "prediction": CLASS_NAMES[predicted_index.item()],
            "confidence": float(confidence.item()) * 100,
            "heatmap_path": str(heatmap_path),
            "overlay_path": str(overlay_path),
        }

    finally:
        forward_handle.remove()
        backward_handle.remove()


if __name__ == "__main__":
    print("Grad-CAM module created.")
    print(f"Model file: {MODEL_PATH}")
    print(f"Model file exists: {MODEL_PATH.exists()}")
    print(f"Device: {DEVICE}")
    print("Next, we will test it with an eye image.")
