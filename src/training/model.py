import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# 2. Load pretrained EfficientNet-B0
# --------------------------------------------------

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)


# --------------------------------------------------
# 3. Replace final classification layer
# --------------------------------------------------

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    4
)


# --------------------------------------------------
# 4. Move model to device
# --------------------------------------------------

model = model.to(device)


# --------------------------------------------------
# 5. Print model information
# --------------------------------------------------

print("\nEfficientNet-B0 loaded successfully.")
print("Number of output classes: 4")
print("Classes:")
print("0 - cataract")
print("1 - diabetic_retinopathy")
print("2 - glaucoma")
print("3 - normal")

print("\nFinal classification layer:")
print(model.classifier)

print("\nModel is ready for training.")