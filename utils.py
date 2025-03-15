import torch
import numpy as np 

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn 

def compute_psnr(pred_masks, masks):
    psnr = PeakSignalNoiseRatio()
    scores = []
    for pred_mask, mask in zip(pred_masks, masks):
        pred_mask = pred_mask.unsqueeze(0)  # Thêm chiều batch
        mask = mask.unsqueeze(0)  # Thêm chiều batch
        score = psnr(pred_mask, mask)
        scores.append(score.item())
    return np.mean(scores)


def compute_ssim (pred_masks, masks):
    ssim = StructuralSimilarityIndexMeasure()
    scores = []
    for pred_mask, mask in zip(pred_masks, masks):
        pred_mask = pred_mask.unsqueeze(0)  # Thêm chiều batch
        mask = mask.unsqueeze(0)  # Thêm chiều batch
        score = ssim(pred_mask, mask)
        scores.append(score.item())
    return np.mean(scores)

def compute_scores(predicted_masks: list, masks: list) -> dict:
    metrics = {
        "psnr": compute_psnr,
        "ssim": compute_ssim
    }

    scores = {metric_name: [] for metric_name in metrics}

    for predicted_mask, mask in zip(predicted_masks, masks):
        for metric_name, scorer in metrics.items():
            scores[metric_name].append(scorer(predicted_mask, mask))

    return {metric_name: np.mean(values) for metric_name, values in scores.items()}

def compute_mse_loss(pred_masks, masks):
    mse_loss = nn.MSELoss()
    losses = []
    for pred_mask, mask in zip(pred_masks, masks):
        loss = mse_loss(pred_mask, mask)
        losses.append(loss) 
    # Calculate the mean of losses using torch.mean()
    return torch.mean(torch.stack(losses))  # Changed: Calculate mean with PyTorch
