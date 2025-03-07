#################################################################################
# project: echocardiogram segmentation of LV, LA and MYO for multiview features
# name: fatima zahra
# date: 07.03.2025
#################################################################################

import os
import cv2
import numpy as np
import nibabel as nib
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.cuda.empty_cache()

# data augmentation

augmentation_transform = A.Compose([
    A.Rotate(limit=5, p=0.2),  
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.2),  
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),  
    A.Normalize(mean=[0.0], std=[1.0]),
    ToTensorV2()
])

# camus data processing

def find_nifti_file(patient_path, base_filename):
    for ext in [".nii.gz", ".nii"]:
        file_path = os.path.join(patient_path, base_filename + ext)
        if os.path.exists(file_path):
            return file_path
    return None

def load_camus_data(camus_dir, view="2CH", target_size=(112, 112)):
    images, masks, filenames = [], [], []
    
    for patient_folder in sorted(os.listdir(camus_dir)):
        patient_path = os.path.join(camus_dir, patient_folder)
        if not os.path.isdir(patient_path):
            continue  # skip non-directory files
        
        patient_id = patient_folder.lower()

        for phase in ["ED", "ES"]:
            img_path = os.path.join(patient_path, f"{patient_id}_{view}_{phase}.nii")
            gt_path = os.path.join(patient_path, f"{patient_id}_{view}_{phase}_gt.nii")

            if os.path.exists(img_path) and os.path.exists(gt_path):
                print(f"[DEBUG] Loading: {img_path} and {gt_path}")

                # load and preprocess image
                try:
                    img = nib.load(img_path).get_fdata()
                    img = img[..., 0] if img.ndim == 3 else img  # ensure 2D format
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    img = (img / 127.5) - 1  # normalise to [-1, 1]
                    print(f"[DEBUG] Image shape: {img.shape}, Min: {img.min()}, Max: {img.max()}")
                except Exception as e:
                    print(f"[ERROR] Failed to load image {img_path}: {e}")
                    continue

                # load and preprocess ground truth mask
                try:
                    gt = nib.load(gt_path).get_fdata()
                    gt = gt[..., 0] if gt.ndim == 3 else gt  
                    gt = np.rint(gt).astype(np.int32)  
                    gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_NEAREST)

                    # debug: check unique values in GT
                    unique_values = np.unique(gt)
                    print(f"[DEBUG] GT unique values: {unique_values}")

                    # ensure only expected labels exist
                    valid_labels = {0, 1, 2, 3}
                    if not set(unique_values).issubset(valid_labels):
                        raise ValueError(f"[ERROR] Unexpected labels in GT mask: {unique_values}")

                    # create multi-channel masks with labels:
                    # LV = 1, LA = 2, MYO = 3
                    lv_mask = (gt == 1).astype(np.float32)
                    la_mask = (gt == 2).astype(np.float32)
                    myo_mask = (gt == 3).astype(np.float32)
                    background = 1 - (lv_mask + la_mask + myo_mask)
                    combined_mask = np.stack([background, lv_mask, la_mask, myo_mask], axis=-1)
                except Exception as e:
                    print(f"[ERROR] Failed to load ground truth {gt_path}: {e}")
                    continue

                images.append(img)
                masks.append(combined_mask)
                filenames.append(f"{patient_id}_{view}_{phase}")
    
    images = np.array(images)
    masks = np.array(masks)
    
    print(f"[DEBUG] Loaded {len(images)} images, {len(masks)} masks, {len(filenames)} filenames")
    return torch.tensor(images).unsqueeze(1), torch.tensor(masks), filenames

# dataset class

class CamusDataset(Dataset):
    def __init__(self, images_2ch, images_4ch, masks, filenames_2ch, filenames_4ch, transform=None):
        self.images_2ch = images_2ch  
        self.images_4ch = images_4ch  
        self.masks = masks  
        self.filenames_2ch = filenames_2ch  
        self.filenames_4ch = filenames_4ch  
        self.transform = transform  

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image_2ch = self.images_2ch[idx]
        image_4ch = self.images_4ch[idx]
        mask = self.masks[idx]
        filename_2ch = self.filenames_2ch[idx]
        filename_4ch = self.filenames_4ch[idx]

        if self.transform:
            image_2ch_np = np.expand_dims(image_2ch.cpu().numpy().squeeze(0), axis=-1)
            image_4ch_np = np.expand_dims(image_4ch.cpu().numpy().squeeze(0), axis=-1)
            mask_np = mask.cpu().numpy()
            
            augmented_2ch = self.transform(image=image_2ch_np, mask=mask_np)
            augmented_4ch = self.transform(image=image_4ch_np, mask=mask_np)
            image_2ch = augmented_2ch['image']
            image_4ch = augmented_4ch['image']
            mask = augmented_2ch['mask']
        
        return image_2ch, image_4ch, mask, filename_2ch, filename_4ch

# loss functions

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, num_classes=4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = torch.clamp(y_true, 0, self.num_classes - 1)
        y_true_cpu = y_true.cpu() if y_true.is_cuda else y_true
        print("[DEBUG] Unique values in y_true:", y_true_cpu.unique())
        
        if (y_true_cpu.min() < 0) or (y_true_cpu.max() >= self.num_classes):
            raise ValueError(f"[ERROR] y_true contains values outside [0, {self.num_classes - 1}]. "
                             f"Min: {y_true_cpu.min().item()}, Max: {y_true_cpu.max().item()}")
        
        y_pred_hard = torch.argmax(y_pred, dim=1)
        total_dice = 0.0
        dice_scores = []
        
        for i in range(1, self.num_classes):
            y_true_c = (y_true == i).float()
            y_pred_c = (y_pred_hard == i).float()
            intersection = torch.sum(y_true_c * y_pred_c)
            denominator = torch.sum(y_true_c) + torch.sum(y_pred_c)
            dice_c = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            dice_scores.append(dice_c.item())
            total_dice += dice_c
        
        print(f"[DEBUG] Dice scores per class: {dice_scores}")
        return 1 - (total_dice / (self.num_classes - 1))

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1, num_classes=4):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.long()
        y_true = torch.argmax(y_true, dim=-1)  # convert one-hot to indices
        ce = self.ce_loss(y_pred, y_true)
        dice = self.dice_loss(y_true, y_pred)
        print(f"[DEBUG] CE Loss: {ce.item()}, Dice Loss: {dice.item()}, Combined Loss: {(self.alpha * ce + (1 - self.alpha) * dice).item()}")
        return self.alpha * ce + (1 - self.alpha) * dice

# morphological cleanup function

def morphological_cleanup(mask, kernel_size=3):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

# attention modules: adaptive directional attention, patchwise self attention & cross attention

class ADAB(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ADAB, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.theta_h = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.theta_w = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        reduced_dim = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
                
    def forward(self, x):
        attn = self.mlp(x)
        x = x * attn
        x_h = x * self.theta_h
        x_w = x * self.theta_w
        x_out = x_h + x_w
        return x_out

class PatchwiseSelfAttention(nn.Module):
    def __init__(self, in_channels, patch_size=4, embed_dim=64, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.attn_scale = nn.Parameter(torch.ones(1, 1, 1))
        self.upsample = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        attn_output, _ = self.self_attn(x, x, x)
        x = (attn_output * self.attn_scale).transpose(1, 2).view(B, self.embed_dim, H // self.patch_size, W // self.patch_size)
        x = self.upsample(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)  # Now using reduction_ratio
        self.query = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.pos_encoding = nn.Parameter(torch.randn(1, reduced_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_q, x_kv):
        batch, C, H, W = x_q.shape
        q = self.query(x_q) + self.pos_encoding
        k = self.key(x_kv) + self.pos_encoding
        v = self.value(x_kv)
        q = q.view(batch, -1, H * W).permute(0, 2, 1)
        k = k.view(batch, -1, H * W).permute(0, 2, 1)
        attn_map = torch.bmm(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        attn_map = F.softmax(attn_map, dim=-1)
        v = v.view(batch, -1, H * W)
        attn_output = torch.bmm(v, attn_map).view(batch, C, H, W)
        return self.gamma * attn_output + x_q

# hybrid cross attention for the fusion of 2 views

class HCAFusion(nn.Module):
    def __init__(self, in_channels):
        super(HCAFusion, self).__init__()
        self.adab_2ch = ADAB(in_channels)
        self.adab_4ch = ADAB(in_channels)
        self.self_attn_2ch = PatchwiseSelfAttention(in_channels)
        self.self_attn_4ch = PatchwiseSelfAttention(in_channels)
        self.cross_attn_2ch = CrossAttention(in_channels)
        self.cross_attn_4ch = CrossAttention(in_channels)
        self.up_conv = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x_2ch, x_4ch):
        x_2ch = self.adab_2ch(x_2ch)
        x_4ch = self.adab_4ch(x_4ch)
        x_2ch = self.self_attn_2ch(x_2ch)
        x_4ch = self.self_attn_4ch(x_4ch)
        x_2ch_ca = self.cross_attn_2ch(x_2ch, x_4ch)
        x_4ch_ca = self.cross_attn_4ch(x_4ch, x_2ch)
        x_fused = x_2ch_ca + x_4ch_ca
        x_fused = self.up_conv(x_fused)
        return x_fused

# squeeze and excitation block for further feature extraction of fused views

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        se_weight = self.global_avg_pool(x).view(batch, channels)
        se_weight = self.fc(se_weight).view(batch, channels, 1, 1)
        return x + (x * se_weight)  

# multiview unet

class MultiViewUNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_p=0.1):
        super(MultiViewUNet, self).__init__()
        # Fusion and SE block
        self.hca_fusion = HCAFusion(in_channels)
        self.se_fusion = SEBlock(64)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        # Define a common pooling layer (kernel=2, stride=2)
        self.pool = nn.MaxPool2d(2)
        
        # Encoder Path
        # Level 1: Already given by SE fusion output (64 channels)
        self.encoder2 = self.conv_block(64, 128)   # Level 2: 64 -> 128
        self.encoder3 = self.conv_block(128, 256)    # Level 3: 128 -> 256
        self.encoder4 = self.conv_block(256, 512)    # Level 4: 256 -> 512
        
        # Bottleneck: from 512 to 1024 channels
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder Path
        # Level 4 Decoder: Up4, concat with encoder4, then decoder4
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)   # 512 (upsampled) + 512 from encoder4
        
        # Level 3 Decoder: Up3, concat with encoder3, then decoder3
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)    # 256 + 256
        
        # Level 2 Decoder: Up2, concat with encoder2, then decoder2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)    # 128 + 128
        
        # Level 1 Decoder: Up1, concat with SE fusion (enc1), then decoder1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)     # 64 + 64
        
        # Final segmentation output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x_2ch, x_4ch):
        # Fusion and SE Block
        x_fused = self.hca_fusion(x_2ch, x_4ch)
        enc1 = self.se_fusion(x_fused)  # enc1: 64 channels
        enc1 = self.dropout(enc1)
        
        # Encoder Path
        x1 = self.pool(enc1)            # Downsample level 1
        enc2 = self.encoder2(x1)        # enc2: 128 channels
        
        x2 = self.pool(enc2)            # Downsample level 2
        enc3 = self.encoder3(x2)        # enc3: 256 channels
        
        x3 = self.pool(enc3)            # Downsample level 3
        enc4 = self.encoder4(x3)        # enc4: 512 channels
        
        x4 = self.pool(enc4)            # Downsample level 4
        x5 = self.bottleneck(x4)        # Bottleneck: 1024 channels
        x5 = self.dropout(x5)
        
        # Decoder Path
        x6 = self.up4(x5)               # Upsample to 512 channels
        x6 = torch.cat([x6, enc4], dim=1)  # Concat with encoder level 4 (512 + 512 = 1024)
        x6 = self.decoder4(x6)          # Output: 512 channels
        
        x7 = self.up3(x6)               # Upsample to 256 channels
        x7 = torch.cat([x7, enc3], dim=1)  # Concat with encoder level 3 (256 + 256 = 512)
        x7 = self.decoder3(x7)          # Output: 256 channels
        
        x8 = self.up2(x7)               # Upsample to 128 channels
        x8 = torch.cat([x8, enc2], dim=1)  # Concat with encoder level 2 (128 + 128 = 256)
        x8 = self.decoder2(x8)          # Output: 128 channels
        
        x9 = self.up1(x8)               # Upsample to 64 channels
        x9 = torch.cat([x9, enc1], dim=1)  # Concat with encoder level 1 (64 + 64 = 128)
        x9 = self.decoder1(x9)          # Output: 64 channels
        
        return self.final_conv(x9)
    
    def adjust_dropout(self, epoch, max_epoch, initial_dropout=0.1, final_dropout=0.01):
        drop_rate = initial_dropout - (initial_dropout - final_dropout) * (epoch / max_epoch)
        # Update dropout rate dynamically in all Dropout2d layers
        for layer in self.modules():
            if isinstance(layer, nn.Dropout2d):
                layer.p = drop_rate


# metrics and graphs

def save_metrics_and_plots(dice_metrics, loss_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "training_metrics_summary.txt")
    
    train_losses, val_losses = loss_metrics
    dice_lv, dice_la, dice_myo, dice_avg = dice_metrics

    mean_train_loss = np.mean(train_losses)
    mean_val_loss = np.mean(val_losses)
    mean_dice_lv = np.mean(dice_lv)
    mean_dice_la = np.mean(dice_la)
    mean_dice_myo = np.mean(dice_myo)
    mean_dice_overall = np.mean(dice_avg)
    
    with open(txt_path, "w") as f:
        f.write("Average Train Loss: {:.4f}\n".format(mean_train_loss))
        f.write("Average Val Loss: {:.4f}\n".format(mean_val_loss))
        f.write("Average Dice LV: {:.4f}\n".format(mean_dice_lv))
        f.write("Average Dice LA: {:.4f}\n".format(mean_dice_la))
        f.write("Average Dice MYO: {:.4f}\n".format(mean_dice_myo))
        f.write("Average Dice Overall: {:.4f}\n".format(mean_dice_overall))
    
    # loss plot
    fig_loss, ax_loss = plt.subplots()
    epochs = range(1, len(train_losses) + 1)
    ax_loss.plot(epochs, train_losses, label="Train Loss", marker='o', color='tab:blue')
    ax_loss.plot(epochs, val_losses, label="Val Loss", marker='x', color='tab:blue', linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Metrics")
    ax_loss.legend(loc='best')
    fig_loss.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_metrics.png"))
    plt.close(fig_loss)
    
    # dice plot
    fig_dice, ax_dice = plt.subplots()
    ax_dice.plot(epochs, dice_lv, label="Dice LV", marker='o', color='tab:green')
    ax_dice.plot(epochs, dice_la, label="Dice LA", marker='x', color='tab:orange')
    ax_dice.plot(epochs, dice_myo, label="Dice MYO", marker='^', color='tab:red')
    ax_dice.plot(epochs, dice_avg, label="Avg Dice", marker='s', color='tab:purple')
    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Dice Coefficient")
    ax_dice.set_title("Dice Metrics")
    ax_dice.legend(loc='best')
    fig_dice.tight_layout()
    plt.savefig(os.path.join(output_dir, "dice_metrics.png"))
    plt.close(fig_dice)

# training and evaluation

EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.0001

camus_dir = "/medical00/alhajjfz/datasets/camus_processed/processed_data"
images_2ch, masks, filenames_2ch = load_camus_data(camus_dir, view="2CH", target_size=(112,112))
images_4ch, _, filenames_4ch = load_camus_data(camus_dir, view="4CH", target_size=(112,112))

patient_ids_2ch = {fname.split("_")[0] for fname in filenames_2ch}
patient_ids_4ch = {fname.split("_")[0] for fname in filenames_4ch}
assert patient_ids_2ch == patient_ids_4ch, "[ERROR] Mismatch in patient IDs between 2CH and 4CH"

dataset = CamusDataset(images_2ch, images_4ch, masks, filenames_2ch, filenames_4ch, transform=augmentation_transform)

total_samples = len(dataset)
train_size = int(0.8 * total_samples)
remaining = total_samples - train_size
val_size = test_size = remaining // 2
train_dataset, temp_dataset = random_split(dataset, [train_size, remaining])
val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiViewUNet(in_channels=1, num_classes=4, dropout_p=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CombinedLoss(num_classes=4).to(device)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses = []
val_losses = []
val_dice_LV = []
val_dice_LA = []
val_dice_MYO = []
val_dice_avg = []

def compute_dice_coefficients(y_true, y_pred, num_classes=4, smooth=1e-6):
    y_pred_labels = torch.argmax(y_pred, dim=1)
    dice_scores = []
    for cls in range(1, num_classes):
        y_true_c = (y_true == cls).float()
        y_pred_c = (y_pred_labels == cls).float()
        intersection = (y_true_c * y_pred_c).sum()
        denominator = y_true_c.sum() + y_pred_c.sum()
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        dice_scores.append(dice.item())
    return dice_scores

# --- training Loop ---
for epoch in range(EPOCHS):
    model.adjust_dropout(epoch, EPOCHS, initial_dropout=0.1, final_dropout=0.01)
    model.train()
    total_loss = 0.0
    for batch_idx, (images_2ch, images_4ch, masks, filenames_2ch, filenames_4ch) in enumerate(train_loader):
        images_2ch = images_2ch.to(device, dtype=torch.float32)
        images_4ch = images_4ch.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)
        optimizer.zero_grad()
        predictions = model(images_2ch, images_4ch)
        loss = criterion(predictions, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # validation
    model.eval()
    val_loss = 0.0
    dice_scores_list = []
    with torch.no_grad():
        for images_2ch, images_4ch, masks, filenames_2ch, filenames_4ch in val_loader:
            images_2ch = images_2ch.to(device, dtype=torch.float32)
            images_4ch = images_4ch.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)
            print(f"[DEBUG] Unique GT labels before loss: {torch.unique(masks).tolist()}")
            predictions = model(images_2ch, images_4ch)
            loss = criterion(predictions, masks)
            val_loss += loss.item()
            masks_labels = torch.argmax(masks, dim=-1)
            dice_scores = compute_dice_coefficients(masks_labels, predictions, num_classes=4)
            dice_scores_list.append(dice_scores)
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    dice_scores_array = np.array(dice_scores_list)
    avg_dice_per_class = np.mean(dice_scores_array, axis=0)
    avg_dice = np.mean(avg_dice_per_class)
    val_dice_LV.append(avg_dice_per_class[0])
    val_dice_LA.append(avg_dice_per_class[1])
    val_dice_MYO.append(avg_dice_per_class[2])
    val_dice_avg.append(avg_dice)
    print(f"[DEBUG] Epoch {epoch+1}/{EPOCHS} - LR: {scheduler.get_last_lr()[0]:.6f} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Dice LV: {avg_dice_per_class[0]:.4f} - Dice LA: {avg_dice_per_class[1]:.4f} - Dice MYO: {avg_dice_per_class[2]:.4f} - Avg Dice: {avg_dice:.4f}")

torch.save(model.state_dict(), "multiview_unet_model.pth")
print("model saved successfully!")

save_metrics_and_plots(dice_metrics=(val_dice_LV, val_dice_LA, val_dice_MYO, val_dice_avg),
                       loss_metrics=(train_losses, val_losses),
                       output_dir="multiview_unet_training_results")

# test evaluations and saving predictions

model.eval()
dice_scores_test = []
combined_losses_test = []
output_test_dir = "multiview_unet_test_results"
os.makedirs(output_test_dir, exist_ok=True)

with torch.no_grad():
    for idx, (img_2ch, img_4ch, mask, filename_2ch, filename_4ch) in enumerate(test_loader):
        if isinstance(filename_2ch, (tuple, list)):
            filename_2ch = filename_2ch[0]
        if isinstance(filename_4ch, (tuple, list)):
            filename_4ch = filename_4ch[0]
        
        parts_2ch = filename_2ch.split("_")
        parts_4ch = filename_4ch.split("_")
        
        patient_id = parts_2ch[0] 
        phase = parts_2ch[2] 
        
        out_name = f"{patient_id}_{phase}"
        
        img_2ch = img_2ch.to(device, dtype=torch.float32)
        img_4ch = img_4ch.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)
        
        prediction = model(img_2ch, img_4ch)
        loss = criterion(prediction, mask)
        combined_losses_test.append(loss.item())
        
        pred_labels = torch.argmax(prediction, dim=1)
        gt_labels = torch.argmax(mask, dim=-1)
        
        pred_labels_np = pred_labels.cpu().squeeze().numpy().astype(np.uint8)
        gt_labels_np = gt_labels.cpu().squeeze().numpy().astype(np.uint8)

        vis_pred = pred_labels_np * 85  
        vis_gt = gt_labels_np * 85

        dice_loss_fn = DiceLoss(num_classes=4)
        gt_labels_tensor = torch.tensor(gt_labels_np, dtype=torch.long).to(device)
        dice_loss_sample = dice_loss_fn(gt_labels_tensor, prediction)
        dice_score = 1 - dice_loss_sample.item()
        dice_scores_test.append(dice_score)
        
        pred_np_clean = morphological_cleanup(vis_pred, kernel_size=3)
        gt_np_clean = morphological_cleanup(vis_gt, kernel_size=3)
        
        cv2.imwrite(os.path.join(output_test_dir, f"{out_name}_pred.png"), pred_np_clean)
        cv2.imwrite(os.path.join(output_test_dir, f"{out_name}_gt.png"), gt_np_clean)
        
        for cls, name in zip([1, 2, 3], ["LV", "MYO", "LA"]):
            pred_cls = ((pred_labels_np == cls).astype(np.uint8)) * 255
            pred_cls_clean = morphological_cleanup(pred_cls, kernel_size=3)
            gt_cls = ((gt_labels_np == cls).astype(np.uint8)) * 255
            gt_cls_clean = morphological_cleanup(gt_cls, kernel_size=3)
            cv2.imwrite(os.path.join(output_test_dir, f"{out_name}_pred_{name}.png"), pred_cls_clean)
            cv2.imwrite(os.path.join(output_test_dir, f"{out_name}_gt_{name}.png"), gt_cls_clean)
        
        overlay = np.zeros((pred_np_clean.shape[0], pred_np_clean.shape[1], 3), dtype=np.uint8)
        color_map = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
        for cls in [1, 2, 3]:
            pred_mask = ((pred_labels_np == cls).astype(np.uint8))
            gt_mask = ((gt_labels_np == cls).astype(np.uint8))
            overlay[pred_mask == 1] = np.array(color_map[cls]) // 2
            overlay[gt_mask == 1] = np.array(color_map[cls])
        cv2.imwrite(os.path.join(output_test_dir, f"{out_name}_overlay.png"), overlay)

save_metrics_and_plots(dice_metrics=(val_dice_LV, val_dice_LA, val_dice_MYO, val_dice_avg),
                       loss_metrics=(train_losses, val_losses),
                       output_dir="multiview_unet_training_results")
print("training and test evaluation complet!")

