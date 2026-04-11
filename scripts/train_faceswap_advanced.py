import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import torchvision.models as models
from torchvision import transforms


# ==================== DATASET ====================
class FaceSwapDataset(Dataset):
    def __init__(self, source_dir, target_dir, img_size=256):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.img_size = img_size
        self.source_images = self._load_image_paths(self.source_dir)
        self.target_images = self._load_image_paths(self.target_dir)
        if len(self.source_images) == 0 or len(self.target_images) == 0:
            raise ValueError("❌ No images found in source or target directory")
        print(f"✅ Loaded {len(self.source_images)} source faces and {len(self.target_images)} target faces")

    def _load_image_paths(self, directory):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        for img_path in directory.rglob('*'):
            if img_path.suffix.lower() in valid_extensions:
                image_paths.append(str(img_path))
        return image_paths

    def _preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __len__(self):
        return len(self.target_images)

    def __getitem__(self, idx):
        target_img = self._preprocess_image(self.target_images[idx])
        source_idx = np.random.randint(0, len(self.source_images))
        source_img = self._preprocess_image(self.source_images[source_idx])
        if target_img is None or source_img is None:
            return self.__getitem__((idx + 1) % len(self))
        return {'source': source_img, 'target': target_img}


# ==================== ATTENTION MODULE ====================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ==================== RESIDUAL BLOCK ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = CBAM(out_channels)
        self.stride = stride

        # Create downsample layer if needed
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)

        # Apply downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ==================== ENHANCED GENERATOR ====================
class EnhancedSwapGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(EnhancedSwapGenerator, self).__init__()

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Encoder with residual blocks - FIX: specify stride and downsample correctly
        self.encoder = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),  # Keep same channels before next stride
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),  # Keep same channels before next stride
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),  # Keep same channels
        )

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )

        # Decoder with transpose convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
        )

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        x = self.initial_conv(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x


# ==================== DISCRIMINATOR ====================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.network(x)


# ==================== PERCEPTUAL LOSS ====================
class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:16]).to(device)
        for param in self.features.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, x, y):
        x_vgg = self.features(x)
        y_vgg = self.features(y)
        return self.mse_loss(x_vgg, y_vgg)


# ==================== IDENTITY LOSS ====================
class IdentityLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(IdentityLoss, self).__init__()
        # Use a pre-trained face recognition model (e.g., FaceNet embeddings)
        # For simplicity, we use VGG features as identity preserving loss
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:30]).to(device)
        for param in self.features.parameters():
            param.requires_grad = False
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, swapped_face, source_face):
        swapped_features = self.features(swapped_face)
        source_features = self.features(source_face)

        # Flatten features
        swapped_flat = swapped_features.view(swapped_features.size(0), -1)
        source_flat = source_features.view(source_features.size(0), -1)

        # Compute cosine similarity
        similarity = self.cosine_similarity(swapped_flat, source_flat)
        # Loss is 1 - similarity (higher similarity = lower loss)
        return 1 - similarity.mean()


# ==================== TRAINER ====================
class FaceSwapTrainerAdvanced:
    def __init__(self, device='cuda', checkpoint_dir='models/checkpoints'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.generator = EnhancedSwapGenerator().to(self.device)
        self.discriminator = PatchDiscriminator().to(self.device)

        # Loss functions
        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss(device=self.device)
        self.criterion_identity = IdentityLoss(device=self.device)

        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Learning rate scheduler
        self.scheduler_g = optim.lr_scheduler.StepLR(self.optimizer_g, step_size=10, gamma=0.5)
        self.scheduler_d = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=10, gamma=0.5)

        self.metrics = {
            'generator_loss': [],
            'discriminator_loss': [],
            'l1_loss': [],
            'perceptual_loss': [],
            'identity_loss': [],
            'adversarial_loss': []
        }

        print(f"✅ Advanced Trainer initialized on device: {self.device}")
        print(f"📊 Generator parameters: {self._count_parameters(self.generator):,}")
        print(f"📊 Discriminator parameters: {self._count_parameters(self.discriminator):,}")

    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_epoch(self, dataloader, epoch):
        epoch_metrics = {
            'generator_loss': [],
            'discriminator_loss': [],
            'l1_loss': [],
            'perceptual_loss': [],
            'identity_loss': [],
            'adversarial_loss': []
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)

            # ===== DISCRIMINATOR TRAINING =====
            self.discriminator.zero_grad()

            real_output = self.discriminator(target)
            loss_d_real = self.criterion_bce(real_output, torch.ones_like(real_output))

            with torch.no_grad():
                fake = self.generator(source, target)

            fake_output = self.discriminator(fake.detach())
            loss_d_fake = self.criterion_bce(fake_output, torch.zeros_like(fake_output))

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            self.optimizer_d.step()

            epoch_metrics['discriminator_loss'].append(loss_d.item())

            # ===== GENERATOR TRAINING =====
            self.generator.zero_grad()

            fake = self.generator(source, target)
            fake_output = self.discriminator(fake)

            # Adversarial loss
            loss_g_adv = self.criterion_bce(fake_output, torch.ones_like(fake_output))

            # L1 loss (pixel-level)
            loss_l1 = self.criterion_l1(fake, target)

            # Perceptual loss (VGG feature matching)
            loss_perceptual = self.criterion_perceptual(fake, target)

            # Identity loss (preserve source face identity)
            loss_identity = self.criterion_identity(fake, source)

            # Combined generator loss
            loss_g = (loss_g_adv +
                      10 * loss_l1 +
                      0.1 * loss_perceptual +
                      0.1 * loss_identity)

            loss_g.backward()
            self.optimizer_g.step()

            epoch_metrics['generator_loss'].append(loss_g.item())
            epoch_metrics['l1_loss'].append(loss_l1.item())
            epoch_metrics['perceptual_loss'].append(loss_perceptual.item())
            epoch_metrics['identity_loss'].append(loss_identity.item())
            epoch_metrics['adversarial_loss'].append(loss_g_adv.item())

            pbar.set_postfix({
                'G': f"{loss_g.item():.4f}",
                'D': f"{loss_d.item():.4f}",
                'L1': f"{loss_l1.item():.4f}",
                'Perc': f"{loss_perceptual.item():.4f}",
                'Id': f"{loss_identity.item():.4f}"
            })

        # Average metrics for epoch
        for key in epoch_metrics:
            avg = np.mean(epoch_metrics[key])
            self.metrics[key].append(avg)

        return epoch_metrics

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
            'metrics': self.metrics
        }
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")

    def train(self, source_dir, target_dir, epochs=50, batch_size=4):
        dataset = FaceSwapDataset(source_dir, target_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        print(f"\n🚀 Starting training for {epochs} epochs on {self.device}")
        print(f"📊 Batch size: {batch_size}, Total batches: {len(dataloader)}\n")

        for epoch in range(epochs):
            epoch_metrics = self.train_epoch(dataloader, epoch)

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

            # Print metrics
            if (epoch + 1) % 1 == 0:
                print(f"\n📊 Epoch {epoch + 1}/{epochs}")
                print(f"  Generator Loss: {self.metrics['generator_loss'][-1]:.4f}")
                print(f"  Discriminator Loss: {self.metrics['discriminator_loss'][-1]:.4f}")
                print(f"  L1 Loss: {self.metrics['l1_loss'][-1]:.4f}")
                print(f"  Perceptual Loss: {self.metrics['perceptual_loss'][-1]:.4f}")
                print(f"  Identity Loss: {self.metrics['identity_loss'][-1]:.4f}")

            # Step learning rate scheduler
            self.scheduler_g.step()
            self.scheduler_d.step()

        print("\n✅ Training completed!")
        self._save_metrics()

    def _save_metrics(self):
        metrics_file = self.checkpoint_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='Train advanced face swap model')
    parser.add_argument('--source-dir', type=str, required=True, help='Source faces directory')
    parser.add_argument('--target-dir', type=str, required=True, help='Target faces directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    trainer = FaceSwapTrainerAdvanced(device=args.device, checkpoint_dir=args.checkpoint_dir)
    trainer.train(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()