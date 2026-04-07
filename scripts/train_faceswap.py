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

class SimpleSwapGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SimpleSwapGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        encoded = self.encoder(x)
        swapped = self.decoder(encoded)
        return swapped

class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class FaceSwapTrainer:
    def __init__(self, device='cuda', checkpoint_dir='models/checkpoints'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.generator = SimpleSwapGenerator().to(self.device)
        self.discriminator = SimpleDiscriminator().to(self.device)
        self.criterion_l1 = nn.L1Loss()
        self.criterion_bce = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.metrics = {'generator_loss': [], 'discriminator_loss': [], 'l1_loss': [], 'identity_similarity': []}
        print(f"✅ Trainer initialized on device: {self.device}")
    
    def train_epoch(self, dataloader, epoch):
        epoch_metrics = {'generator_loss': [], 'discriminator_loss': [], 'l1_loss': []}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
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
            self.generator.zero_grad()
            fake = self.generator(source, target)
            fake_output = self.discriminator(fake)
            loss_g_adv = self.criterion_bce(fake_output, torch.ones_like(fake_output))
            loss_l1 = self.criterion_l1(fake, target)
            loss_g = loss_g_adv + 10 * loss_l1
            loss_g.backward()
            self.optimizer_g.step()
            epoch_metrics['generator_loss'].append(loss_g.item())
            epoch_metrics['l1_loss'].append(loss_l1.item())
            pbar.set_postfix({'G_loss': loss_g.item(), 'D_loss': loss_d.item(), 'L1': loss_l1.item()})
        for key in epoch_metrics:
            avg = np.mean(epoch_metrics[key])
            self.metrics[key].append(avg)
        return epoch_metrics
    
    def save_checkpoint(self, epoch):
        checkpoint = {'epoch': epoch, 'generator_state': self.generator.state_dict(), 'discriminator_state': self.discriminator.state_dict(), 'optimizer_g_state': self.optimizer_g.state_dict(), 'optimizer_d_state': self.optimizer_d.state_dict(), 'metrics': self.metrics}
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def train(self, source_dir, target_dir, epochs=50, batch_size=4):
        dataset = FaceSwapDataset(source_dir, target_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"\n🚀 Starting training for {epochs} epochs on {self.device}")
        for epoch in range(epochs):
            epoch_metrics = self.train_epoch(dataloader, epoch)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
            if (epoch + 1) % 1 == 0:
                print(f"\n📊 Epoch {epoch + 1}/{epochs}")
                print(f"  Generator Loss: {self.metrics['generator_loss'][-1]:.4f}")
                print(f"  Discriminator Loss: {self.metrics['discriminator_loss'][-1]:.4f}")
                print(f"  L1 Loss: {self.metrics['l1_loss'][-1]:.4f}")
        print("✅ Training completed!")
        self._save_metrics()
    
    def _save_metrics(self):
        metrics_file = self.checkpoint_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to: {metrics_file}")

def main():
    parser = argparse.ArgumentParser(description='Train face swap model')
    parser.add_argument('--source-dir', type=str, required=True, help='Source faces directory')
    parser.add_argument('--target-dir', type=str, required=True, help='Target faces directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints', help='Checkpoint directory')
    args = parser.parse_args()
    trainer = FaceSwapTrainer(device=args.device, checkpoint_dir=args.checkpoint_dir)
    trainer.train(source_dir=args.source_dir, target_dir=args.target_dir, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    main()