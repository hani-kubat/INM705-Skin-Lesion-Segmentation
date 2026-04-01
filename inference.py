import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Models import build_model

def load_model(condition, checkpoint_dir='checkpoints'):
    model = build_model(condition)
    checkpoint_path = f'{checkpoint_dir}/best_model_condition_{condition}.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Condition {condition} loaded - Best Val Dice: {checkpoint["val_dice"]:.4f} (epoch {checkpoint["epoch"]})')
    return model

def run_inference(models_dict, val_img_dir, val_mask_dir, num_samples=5):
    val_images = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.jpg')])
    val_masks = sorted([f for f in os.listdir(val_mask_dir) if f.endswith('.png')])

    inference_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models_dict.values():
        model.eval()
        model.to(device)

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, num_samples * 4))
    columns = ['Original Image', 'Ground Truth', 'Condition A\n(Frozen)',
               'Condition B\n(Fine-tuned)', 'Condition C\n(Random)']
    for ax, col in zip(axes[0], columns):
        ax.set_title(col, fontsize=13, fontweight='bold', pad=10)

    for i in range(num_samples):
        img_path = os.path.join(val_img_dir, val_images[i])
        mask_path = os.path.join(val_mask_dir, val_masks[i])
        original_img = np.array(Image.open(img_path).convert('RGB'))
        true_mask = np.array(Image.open(mask_path).convert('L'))
        true_mask_binary = (true_mask > 127).astype(np.float32)
        true_mask_resized = np.array(Image.fromarray(true_mask_binary).resize((256, 256), Image.NEAREST))

        transformed = inference_transform(image=original_img, mask=true_mask_binary)
        input_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_A = torch.sigmoid(models_dict['A'](input_tensor)).squeeze().cpu().numpy()
            pred_B = torch.sigmoid(models_dict['B'](input_tensor)).squeeze().cpu().numpy()
            pred_C = torch.sigmoid(models_dict['C'](input_tensor)).squeeze().cpu().numpy()

        pred_A_binary = (pred_A > 0.5).astype(np.float32)
        pred_B_binary = (pred_B > 0.5).astype(np.float32)
        pred_C_binary = (pred_C > 0.5).astype(np.float32)

        def single_dice(pred, target):
            intersection = (pred * target).sum()
            return (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

        dice_A = single_dice(pred_A_binary, true_mask_resized)
        dice_B = single_dice(pred_B_binary, true_mask_resized)
        dice_C = single_dice(pred_C_binary, true_mask_resized)

        axes[i, 0].imshow(original_img)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(true_mask_resized, cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(pred_A_binary, cmap='gray')
        axes[i, 2].set_xlabel(f'Dice: {dice_A:.4f}', fontsize=10)
        axes[i, 2].axis('off')
        axes[i, 3].imshow(pred_B_binary, cmap='gray')
        axes[i, 3].set_xlabel(f'Dice: {dice_B:.4f}', fontsize=10)
        axes[i, 3].axis('off')
        axes[i, 4].imshow(pred_C_binary, cmap='gray')
        axes[i, 4].set_xlabel(f'Dice: {dice_C:.4f}', fontsize=10)
        axes[i, 4].axis('off')

    plt.suptitle('Inference Results: All 3 Conditions vs Ground Truth',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('inference_results.png', bbox_inches='tight', dpi=150)
    print('Figure saved!')
    plt.show()


if __name__ == '__main__':
    checkpoint_dir = 'checkpoints'
    val_img_dir = 'data/ISIC2018_Task1-2_Validation_Input'
    val_mask_dir = 'data/ISIC2018_Task1_Validation_GroundTruth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_A = load_model('A', checkpoint_dir).to(device)
    model_B = load_model('B', checkpoint_dir).to(device)
    model_C = load_model('C', checkpoint_dir).to(device)

    models_dict = {'A': model_A, 'B': model_B, 'C': model_C}
    run_inference(models_dict, val_img_dir, val_mask_dir, num_samples=5)
