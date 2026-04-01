import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(condition='A'):
    """
    Build U-Net model based on experimental condition.
    Condition A: Pretrained ResNet-34 encoder, FROZEN
    Condition B: Pretrained ResNet-34 encoder, FINE-TUNED
    Condition C: Random weights, no pretraining
    """
    if condition == 'A' or condition == 'B':
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
        if condition == 'A':
            for param in model.encoder.parameters():
                param.requires_grad = False
            print('Condition A: Pretrained encoder FROZEN')
        else:
            print('Condition B: Pretrained encoder FINE-TUNED')
    else:
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        print('Condition C: Random weights, no pretraining')

    return model


class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()
