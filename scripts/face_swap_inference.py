import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


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


class FaceSwapInference:
    def __init__(self, model_path, device='cpu', img_size=256):
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.generator = SimpleSwapGenerator().to(self.device)
        self._load_model(model_path)
        self.generator.eval()

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'generator_state' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator_state'])
        else:
            self.generator.load_state_dict(checkpoint)
        print(f"✅ Model loaded from: {model_path}")

    def _preprocess(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Normalize to [-1, 1] to match the Tanh output range of the generator
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess(self, tensor):
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def swap_faces(self, source_path, target_path, output_path):
        source_tensor = self._preprocess(source_path)
        target_tensor = self._preprocess(target_path)

        with torch.no_grad():
            swapped_tensor = self.generator(source_tensor, target_tensor)

        swapped_image = self._postprocess(swapped_tensor)

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        success = cv2.imwrite(output_path, swapped_image)
        if not success:
            raise IOError(f"cv2.imwrite failed to write image to: {output_path}")

        if not os.path.exists(output_path):
            raise IOError(f"Image was not saved to disk: {output_path}")

        print(f"✅ Swapped image saved to: {output_path}")
        return swapped_image, output_path
