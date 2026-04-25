# INM705 Skin Lesion Segmentation

MSc Artificial Intelligence - City St George's, University of London

## Project Overview
U-Net and UNet++ architectures with ResNet-34 encoder for semantic segmentation of skin lesions using the ISIC 2018 dataset. Four experimental conditions are evaluated to investigate the effect of encoder initialisation strategy and architectural design on segmentation performance.

## Four Experimental Conditions

| Condition | Architecture | Encoder | Best Val Dice |
|---|---|---|---|
| A | U-Net | Pretrained ResNet-34, Frozen | 0.8808 |
| B | U-Net | Pretrained ResNet-34, Fine-tuned | 0.9124 |
| C | U-Net | Random weights | 0.8913 |
| D | UNet++ + scSE Attention | Pretrained ResNet-34, Fine-tuned | 0.9003 |

## Dataset
ISIC 2018 Task 1 - Lesion Segmentation
https://challenge.isic-archive.com/data/#2018

## Project Structure
- `Dataset.py` - ISIC dataset class and augmentation transforms
- `Models.py` - U-Net and UNet++ models, loss function and metrics
- `train.py` - Training pipeline with wandb logging
- `INM705_Inference.ipynb` - Load checkpoints and visualise predictions
- `requirements.txt` - Dependencies

## Training
Run training with:
python train.py --condition A --epochs 30 --train_img_dir data/train/images --train_mask_dir data/train/masks --val_img_dir data/val/images --val_mask_dir data/val/masks

Condition options: A, B, C or D

## Requirements
See requirements.txt. Install with:
pip install -r requirements.txt

## Experiment Tracking
All runs logged via Weights and Biases. Project: INM705-Skin-Lesion-Segmentation

## Model Checkpoints

Trained model weights for all four experimental conditions are saved and available upon request. Checkpoint files are named:

- `best_model_condition_A.pth` — U-Net, frozen pretrained encoder (Val Dice: 0.8808)
- `best_model_condition_B.pth` — U-Net, fine-tuned pretrained encoder (Val Dice: 0.9124)
- `best_model_condition_C.pth` — U-Net, random initialisation (Val Dice: 0.8913)
- `best_model_condition_D.pth` — UNet++ with scSE attention (Val Dice: 0.9003)

Checkpoints were saved during training using `torch.save()` and can be loaded using the `build_model()` function in `inference.ipynb`.
