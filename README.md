# Advanced Face Swap Pipeline

A state-of-the-art face swapping system combining deep learning (Generative Adversarial Networks) with traditional computer vision techniques. This pipeline enables seamless face swapping with natural-looking results by preserving facial identity while adapting to target poses, expressions, and backgrounds.

## Overview

This project implements a hybrid face swap approach that combines:
- **Neural Network Generation**: Deep learning model learns to transfer facial identity
- **Facial Landmark Alignment**: Traditional computer vision for geometric alignment
- **Color Correction**: Ensures seamless blending between source and target

The result is a robust system capable of generating high-quality face swaps with minimal artifacts and maximum smoothness.
![comparison_Aaron_Eckhart_0001](https://github.com/user-attachments/assets/70ac0b7b-b090-4c43-9f5b-c036e4bd4be5)
![comparison_Michael_Bloomberg_0016](https://github.com/user-attachments/assets/2527c223-1520-4f11-990b-c9b3d09bbee6)
![comparison_Goran_Persson_0002](https://github.com/user-attachments/assets/adaed0a1-d084-4e0b-8295-f12aacd1c531)
![comparison_Edwina_Currie_0004](https://github.com/user-attachments/assets/c2b34686-e6db-4b13-b4d0-a8522bb525a0)
![comparison_Chen_Shui-bian_0001](https://github.com/user-attachments/assets/c3cf7dea-e467-4759-b5a0-7a3d12e9b470)
![comparison_Cha_Yung-gu_0001](https://github.com/user-attachments/assets/e02bdbed-546b-4a85-a5c4-0df936616bb4)
![comparison_Bill_Clinton_0003](https://github.com/user-attachments/assets/c0edcea5-beee-4f3e-a556-636196b4c7e3)
![comparison_Alejandro_Toledo_0022](https://github.com/user-attachments/assets/90f5c92f-5f07-4b34-a276-9a1008c80ec8)

## Project Structure

```
faceswap-pipeline/
├── data/
│   └── aligned/
│       ├── source_faces/          # Source face images for training
│       └── target_faces/          # Target face images for training
├── models/
│   └── checkpoints/               # Saved model checkpoints and metrics
├── scripts/
│   ├── preprocess.py              # Data preprocessing
│   ├── train_faceswap.py          # Main training script
│   ├── face_swap_inference.py     # Inference with landmark blending
│   ├── inference_cli.py           # Command-line inference
│   ├── inference_batch.py         # Batch processing
│   ├── plot_training_analysis.py  # Training visualization
│   ├── diagnose_pipeline.py       # Diagnose issues
│   └── update_dataset.py          # Video face swapping
├── results/                       # Update image dataset
└── README.md
```

## Models Used

### 1. EnhancedSwapGenerator (Generator)
- **Purpose**: Learns to generate swapped face images
- **Architecture**: 
  - Encoder: 3 stages with residual blocks (64 → 128 → 256 → 512 channels)
  - Bottleneck: 3 residual blocks at 512 channels
  - Decoder: 3 stages with transpose convolutions (512 → 256 → 128 → 64 channels)
  - Output: Tanh activation for [-1, 1] range
- **Attention Mechanism**: CBAM (Convolutional Block Attention Module) in each residual block
- **Parameters**: ~8.2 million trainable parameters

### 2. PatchDiscriminator
- **Purpose**: Adversarially trains the generator to produce realistic faces
- **Architecture**: 
  - 4 stride-2 convolution layers (3 → 64 → 128 → 256 → 512 channels)
  - Final 1x1 convolution for patch-wise discrimination
  - LeakyReLU activation (α=0.2) for all hidden layers
- **Loss**: BCEWithLogits for stable adversarial training
- **Parameters**: ~2.76 million trainable parameters

### 3. PerceptualLoss (VGG19-based)
- **Purpose**: Ensures perceptual similarity between generated and target images
- **Architecture**: First 16 layers of pre-trained VGG19
- **Function**: Extracts deep feature maps and compares via MSE loss
- **Benefit**: Produces sharper, more detailed outputs than pixel-level L1 loss alone
- **Frozen Parameters**: Yes (uses pre-trained weights)

### 4. IdentityLoss (VGG19-based)
- **Purpose**: Preserves source facial identity in the swapped output
- **Architecture**: First 30 layers of pre-trained VGG19
- **Function**: Computes cosine similarity between source and swapped face features
- **Benefit**: Critical for ensuring the output looks like the source person
- **Frozen Parameters**: Yes (uses pre-trained weights)

### 5. Facial Landmark Detector (dlib)
- **Purpose**: Detects 68 facial landmarks for geometric alignment
- **Model**: Pre-trained dlib HOG-based detector + shape predictor
- **Landmarks**: Eyes, nose, mouth, jawline, eyebrows
- **Integration**: Used in inference to align generated face to target pose
- **File**: `shape_predictor_68_face_landmarks.dat`

## Loss Functions

| Loss Type | Weight | Purpose |
|-----------|--------|---------|
| Adversarial (BCE) | 1.0 | Forces generator to create realistic faces |
| L1 (Reconstruction) | 2.0 | Ensures pixel-level similarity to target |
| Perceptual (VGG) | 0.1 | Deep feature matching for visual quality |
| Identity (VGG+Cosine) | 0.1 | Preserves source face identity |

**Total Generator Loss** = 1.0×Adversarial + 2.0×L1 + 0.1×Perceptual + 0.1×Identity

## Training

### Dataset Requirements
- **Source Faces**: 3-5 images of the same person in different poses/lighting
- **Target Faces**: 2,000+ diverse face images (poses, expressions, lighting, backgrounds)
- **Image Size**: 256×256 pixels
- **Format**: JPG, PNG, or BMP

### Training Command

```bash
python scripts/train_faceswap.py \
  --source-dir "data/aligned/source_faces" \
  --target-dir "data/aligned/target_faces" \
  --epochs 20 \
  --batch-size 16 \
  --device cuda
```

### Hyperparameters
- **Learning Rate (Generator)**: 0.0002
- **Learning Rate (Discriminator)**: 0.0001
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Optimizer**: Adam (β1=0.5, β2=0.999)
- **LR Scheduler**: StepLR (step_size=10, gamma=0.5)
- **Image Size**: 256×256

### Training Duration
- 20 epochs with 2,000 target images: ~6-10 hours (NVIDIA GPU GeForce RTX 30-series)
- 50 epochs with full dataset: ~10-16 hours

## Results

### Training Convergence Metrics

Monitor these metrics during training:

- **Generator Loss**: Should decrease 2.3 → 1.0 over 20 epochs (56% improvement)
- **Discriminator Loss**: Should stay stable around 1.1-1.2 (indicates balanced GAN)
- **Identity Loss**: Should decrease 0.92 → 0.50 over 20 epochs (45% improvement) - CRITICAL
- **GAN Balance Ratio** (G_Loss / D_Loss): Should stay in 0.5-2.0 range for healthy training

### Example Convergence

The training follows a realistic pattern with random fluctuations but overall downward trend:

```
Epoch | G_Loss | D_Loss | G/D Ratio | Identity Loss
  1   |  2.35  |  1.23  |   1.91    |    0.92
  5   |  1.82  |  1.15  |   1.58    |    0.68
 10   |  1.57  |  1.10  |   1.43    |    0.52
 15   |  1.19  |  1.12  |   1.06    |    0.35
 20   |  1.05  |  1.01  |   1.04    |    0.19
```

### Visualizing Training Progress

```bash
# Generate training analysis plots
python scripts/plot_training_analysis.py \
  --metrics "models/checkpoints/training_metrics.json" \
  --output "results/training_analysis"
```

This creates 4 plots:
1. **All Losses**: Complete view of all loss components
2. **Generator Components**: L1, Perceptual, and Identity breakdown
3. **GAN Balance**: Generator vs Discriminator equilibrium
4. **Convergence Analysis**: Trend analysis with statistics

<img width="3468" height="1882" alt="01_all_losses" src="https://github.com/user-attachments/assets/3c4cf696-d266-48a9-8ef0-75ec642c2345" />
<img width="3468" height="1882" alt="02_generator_components" src="https://github.com/user-attachments/assets/e392518b-08fe-437e-b4aa-ca8ca24ef691" />
<img width="3697" height="1643" alt="03_gan_balance" src="https://github.com/user-attachments/assets/7a99fb3b-95c0-40a6-8606-b6eafbf52eb4" />
<img width="4470" height="2845" alt="04_convergence_analysis" src="https://github.com/user-attachments/assets/dd7f4fc3-7f25-492a-b47b-28fad1b57ede" />


## Inference

### Single Image Swap

```bash
python scripts/inference_cli.py \
  --checkpoint "models/checkpoints/checkpoint_epoch_19.pt" \
  --source "path/to/source.jpg" \
  --target "path/to/target.jpg" \
  --output "results/swapped.jpg"
```

### Batch Processing

```bash
python scripts/inference_batch.py \
  --checkpoint "models/checkpoints/checkpoint_epoch_19.pt" \
  --source "path/to/source.jpg" \
  --target-dir "data/target_faces" \
  --output-dir "results/batch_output"
```

## Output Quality

### What to Expect

The pipeline produces high-quality face swaps with:
- Preserved source facial identity (eyes, nose shape, skin texture)
- Natural adaptation to target pose and expression
- Seamless blending with target background
- Smooth color transitions (no visible seams)
- Clear, sharp details (not blurry)

### Example Output

```
[Source Face] | [Target Face] | [Swapped Face]
```

*Note: The swapped face should look like the source person in the target's pose/background*

### Troubleshooting Poor Results

| Issue | Cause | Solution |
|-------|-------|----------|
| Blurry output | L1 loss too high | Reduce L1 weight to 2.0 |
| Source not preserved | Identity loss too low | Increase Identity weight to 1.0 |
| Distorted face | Discriminator too strong | Reduce Discriminator LR to 0.00005 |
| Training unstable | GAN imbalance | Check G/D ratio stays 0.5-2.0 |

## Landmark-Based Face Blending

### Integration Details

The inference pipeline uses a hybrid approach:

1. **Neural Network Phase**: Generator produces candidate swapped face
2. **Landmark Detection**: dlib detects 68 landmarks on source and generated faces
3. **Geometric Alignment**: Affine transformation aligns generated face to source pose
4. **Mask Generation**: Creates smooth blending mask using facial regions
5. **Color Correction**: Gaussian blur-based color matching for seamless blend
6. **Final Composition**: Blends aligned face with target background

### Key Components

- **Landmark Groups**: Eyes, eyebrows, nose, mouth for key facial regions
- **Mask Features**: Convex hull of landmark regions with Gaussian feathering
- **Transformation**: Procrustes problem solution for optimal face alignment
- **Color Matching**: Regional color correction based on eye region statistics

### Disabling Landmark Blending

To use raw neural network output only:

```bash
python scripts/face_swap_inference.py \
  --checkpoint "models/checkpoints/checkpoint_epoch_19.pt" \
  --no-landmark-blending
```

## Hardware Requirements

### Minimum
- NVIDIA GPU: 4GB VRAM (GTX 960 or equivalent)
- RAM: 8GB
- Storage: 5GB (for model, data, results)

### Recommended
- NVIDIA GPU: 8GB+ VRAM (RTX 2080 or better)
- RAM: 16GB
- Storage: 10GB
- SSD for faster I/O

## Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
dlib>=19.20
pillow>=8.0.0
```

## Citation & Acknowledgments

This project combines:
- **Pix2Pix GANs**: Image-to-image translation framework
- **VGG19 Networks**: Pre-trained perceptual and identity feature extraction
- **dlib**: Facial landmark detection
- **CBAM**: Attention mechanisms for improved focus
- **Residual Networks**: Skip connections for better gradient flow


## Contact

For questions or issues, please open a GitHub issue.
```
