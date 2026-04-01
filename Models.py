import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate module as proposed in Oktay et al. (2018)
    Attention U-Net: Learning Where to Look for the Pancreas
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_resized = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g_resized)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


def build_model(condition='A'):
    """
    Build model based on experimental condition.
    Condition A: Pretrained ResNet-34 encoder, FROZEN
    Condition B: Pretrained ResNet-34 encoder, FINE-TUNED
    Condition C: Random weights, no pretraining
    Condition D: UNet++ with scSE attention, pretrained, fine-tuned
    """
    if condition == 'A':
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
        for param in model.encoder.parameters():
            param.requires_grad = False
        print('Condition A: Pretrained encoder FROZEN')

    elif condition == 'B':
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
        print('Condition B: Pretrained encoder FINE-TUNED')

    elif condition == 'C':
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        print('Condition C: Random weights, no pretraining')

    elif condition == 'D':
        model = smp.UnetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type='scse'
        )
        print('Condition D: UNet++ with scSE attention, pretrained fine-tuned')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable params: {trainable:,} / Total: {total:,}')
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
