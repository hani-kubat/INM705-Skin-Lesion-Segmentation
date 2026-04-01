# INM705 Skin Lesion Segmentation

MSc Artificial Intelligence - City St George's, University of London

## Project Overview
U-Net with ResNet-34 encoder for semantic segmentation of skin lesions using the ISIC 2018 dataset.

## Three Experimental Conditions
- **Condition A**: Pretrained ResNet-34 encoder, frozen (Best Val Dice: 0.8808)
- **Condition B**: Pretrained ResNet-34 encoder, fine-tuned (Best Val Dice: 0.9124)
- **Condition C**: Random weights, no pretraining (Best Val Dice: 0.8913)

## Dataset
ISIC 2018 Task 1 - Lesion Segmentation
https://challenge.isic-archive.com/data/#2018

## Project Structure
- `Dataset.py` - ISIC dataset class and augmentation transforms
- `Models.py` - U-Net model, loss function and metrics
- `train.py` - Training pipeline with wandb logging
- `inference.py` - Load checkpoints and visualise predictions
- `requirements.txt` - Dependencies

## Training
Run training with:
```
python train.py --condition A --epochs 30 --train_img_dir data/train/images --train_mask_dir data/train/masks --val_img_dir data/val/images --val_mask_dir data/val/masks
```
