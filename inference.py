import torch
import matplotlib.pyplot as plt
import numpy as np 
from unetpp_model import UNET_PP


def load_model(checkpoint_dir: str, device: torch.device, model=UNET_PP(1, 3)) -> torch.nn.Module:

    checkpoint = torch.load(checkpoint_dir, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print("Model has been loaded successfully!")

    return model


def inference(image: torch.Tensor, mask: torch.Tensor, model: torch.nn.Module, device: torch.device):
    image = image.to(device)
    mask = mask.to(device)

    image = image.unsqueeze(0)

    with torch.no_grad():
        logits = model(image)
     

    image = image.squeeze(0).cpu().permute(1, 2, 0).numpy() # Convert from (C, H, W) to (H, W, C)
    mask = mask.squeeze(0).cpu().permute(1, 2, 0).numpy() # Convert from (C, H, W) to (H, W, C)
    logits = logits.squeeze(0).cpu().permute(1, 2, 0).numpy()
    return image, mask, logits



def plot_results(image, mask, pred):

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Gray Image")
    # image = np.clip(image, 0, 1)
    plt.imshow(image) # image shape is (H, W, 1) - Correct for grayscale
    plt.axis("off")

    # # True mask
    plt.subplot(1, 3, 2)
    plt.title("True RGB Image")
    # mask = np.clip(mask, 0, 1)
    plt.imshow(mask)  # Changed to transpose the mask to (H, W, 3)
    plt.axis("off")

    # Prediction mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted RGB Image")
    # pred = np.clip(pred, 0, 1)
    plt.imshow(pred) # Changed to transpose the prediction to (H, W, 3)

    plt.show()
