from torch.optim import AdamW
from utils import *
from unetpp_model import UNET_PP

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import FlowerDataset

from shutil import copyfile

import os
import torch
import torch.nn as nn

def train_model(epoch: int, model: nn.Module, dataloader: DataLoader, optim: torch.optim.Optimizer, 
                                                            device: torch.device, criterion: nn.L1Loss):
    model.train()

    running_loss = .0
    with tqdm(desc='Epoch %d - Training' % epoch, unit='it', total=len(dataloader)) as pb:
        for it, batch in enumerate(dataloader):
            gray_image = batch['gray_scale'].to(device)
            rgb_image = batch['rgb_scale'].to(device)

            logits = model(gray_image)
            loss = criterion(logits, rgb_image)

            # print(logits)
            
            # Back propagation
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()

            # Update training status
            pb.set_postfix(loss=running_loss / (it + 1))
            pb.update()


def evaluate_model(epoch: int, model: nn.Module, dataloader: DataLoader, 
                                        device: torch.device, criterion: nn.L1Loss) -> dict:
    model.eval()

    all_predictions = []
    all_masks = []

    with tqdm(desc='Epoch %d - Evaluating' % epoch, unit='it', total=len(dataloader)) as pb:
        for batch in dataloader:
            image = batch['gray_scale'].to(device)
            mask = batch['rgb_scale'].to(device)

            with torch.no_grad():
                logits = model(image)

            # probs = torch.sigmoid(logits)
            logits = logits.cpu()
            mask = mask.cpu() 

            all_predictions.append(logits)
            all_masks.append(mask) 

            pb.update()

    scores = compute_scores(all_predictions, all_masks)
    return scores

def save_checkpoint(dict_to_save: dict, checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(dict_to_save, os.path.join(f"{checkpoint_dir}", "last_model.pth"))


def main(checkpoint_dir, train_dir, val_dir, test_dir):
    
    train_image_dir = [] 
    for folder in os.listdir(train_dir):
        for image in os.listdir(os.path.join(train_dir, folder)):
            train_image_dir.append(os.path.join(train_dir, folder, image))

    val_image_dir = []
    for folder in os.listdir(val_dir):
        for image in os.listdir(os.path.join(val_dir, folder)):
            val_image_dir.append(os.path.join(val_dir, folder, image))

    test_image_dir = []
    for image in os.listdir(test_dir): 
        image_dir = os.path.join(test_dir, image)
        test_image_dir.append(image_dir)


    train_dataset = FlowerDataset(train_image_dir)
    val_dataset = FlowerDataset(val_image_dir)
    test_dataset = FlowerDataset(test_image_dir)


    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare for training
    epoch = 0
    allowed_patience = 5
    best_score = 0
    compared_score = "ssim"
    patience = 0
    exit_train = False

    # Define model
    model = UNET_PP(3, 3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    # Training model
    while True:
        train_model(epoch, model, train_loader, optimizer, device, criterion)
        # validate
        scores = evaluate_model(epoch, model, eval_loader, device, criterion)
        print(f"PSNNR: {scores['psnr']}; SSIM: {scores['ssim']}")
        score = scores[compared_score]

        # Prepare for next epoch
        is_best_model = False
        if score > best_score:
            best_score = score
            patience = 0
            is_best_model = True
        else:
            patience += 1

        if patience == allowed_patience:
            exit_train = True


        save_checkpoint({
            "epoch": epoch,
            "best_score": best_score,
            "patience": patience,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, checkpoint_dir)


        if is_best_model:
            copyfile(
                os.path.join(checkpoint_dir, "last_model.pth"),
                os.path.join(checkpoint_dir, "best_model.pth")
            )

        if exit_train or epoch == 2:
            break

        epoch += 1

        
        

if __name__ == "__main__":
    main (
        checkpoint_dir='/home/huatansang/Documents/UnetPP/unetpp_checkpoint', 
        train_dir='/home/huatansang/Documents/UnetPP/dataset/train', 
        val_dir='/home/huatansang/Documents/UnetPP/dataset/valid', 
        test_dir='/home/huatansang/Documents/UnetPP/dataset/test'
    )