import os
import torch
import wandb
from torch.utils.data import DataLoader
from Dataset import ISICDataset, train_transform, val_transform
from Models import build_model, BCEDiceLoss, dice_score, iou_score
import argparse

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(predictions.detach(), masks)
        total_iou += iou_score(predictions.detach(), masks)
    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            loss = criterion(predictions, masks)
            total_loss += loss.item()
            total_dice += dice_score(predictions, masks)
            total_iou += iou_score(predictions, masks)
    n = len(loader)
    return total_loss/n, total_dice/n, total_iou/n


def run_experiment(condition, num_epochs, train_loader, val_loader, checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nStarting Condition {condition} on {device}')

    model = build_model(condition).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    criterion = BCEDiceLoss()
    os.makedirs(checkpoint_dir, exist_ok=True)

    run = wandb.init(
        project='INM705-Skin-Lesion-Segmentation',
        name=f'Condition_{condition}',
        config={
            'condition': condition,
            'encoder': 'resnet34',
            'encoder_frozen': condition == 'A',
            'pretrained': condition != 'C',
            'epochs': num_epochs,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'image_size': 256,
            'loss': 'BCE+Dice',
            'optimizer': 'Adam'
        }
    )

    best_val_dice, best_epoch = 0, 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_dice': train_dice, 'train_iou': train_iou,
            'val_loss': val_loss, 'val_dice': val_dice, 'val_iou': val_iou,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        print(f'  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}')
        print(f'  Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}')

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'condition': condition,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, f'{checkpoint_dir}/best_model_condition_{condition}.pth')
            print(f'  ✓ New best model saved (Val Dice: {best_val_dice:.4f})')

    print(f'\nCondition {condition} complete! Best Val Dice: {best_val_dice:.4f} at epoch {best_epoch}')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='A', choices=['A', 'B', 'C'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_img_dir', type=str, required=True)
    parser.add_argument('--train_mask_dir', type=str, required=True)
    parser.add_argument('--val_img_dir', type=str, required=True)
    parser.add_argument('--val_mask_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    train_dataset = ISICDataset(args.train_img_dir, args.train_mask_dir, train_transform)
    val_dataset = ISICDataset(args.val_img_dir, args.val_mask_dir, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    run_experiment(args.condition, args.epochs, train_loader, val_loader, args.checkpoint_dir)
