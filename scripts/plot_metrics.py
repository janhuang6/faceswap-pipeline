import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np

class MetricsVisualizer:
    def __init__(self, metrics_file):
        self.metrics_file = Path(metrics_file)
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        with open(self.metrics_file, 'r') as f:
            self.metrics = json.load(f)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
    
    def plot_losses(self, save_path='outputs/loss_curves.png'):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(self.metrics['generator_loss']) + 1)
        axes[0].plot(epochs, self.metrics['generator_loss'], 'b-', linewidth=2, label='Generator Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Generator Loss Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].plot(epochs, self.metrics['discriminator_loss'], 'r-', linewidth=2, label='Discriminator Loss')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Discriminator Loss Over Time', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[2].plot(epochs, self.metrics['l1_loss'], 'g-', linewidth=2, label='L1 Loss')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('L1 Reconstruction Loss Over Time', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Loss curves saved to: {save_path}")
        plt.close()
    
    def plot_combined_losses(self, save_path='outputs/combined_losses.png'):
        fig, ax = plt.subplots(figsize=(12, 7))
        epochs = range(1, len(self.metrics['generator_loss']) + 1)
        ax.plot(epochs, self.metrics['generator_loss'], 'b-', linewidth=2.5, label='Generator Loss', marker='o', markersize=4, markevery=5)
        ax.plot(epochs, self.metrics['discriminator_loss'], 'r-', linewidth=2.5, label='Discriminator Loss', marker='s', markersize=4, markevery=5)
        ax.plot(epochs, self.metrics['l1_loss'], 'g-', linewidth=2.5, label='L1 Loss', marker='^', markersize=4, markevery=5)
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=13, fontweight='bold')
        ax.set_title('Face Swap Training - Combined Loss Curves', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Combined loss plot saved to: {save_path}")
        plt.close()
    
    def plot_identity_similarity(self, save_path='outputs/identity_similarity.png'):
        if 'identity_similarity' not in self.metrics or len(self.metrics['identity_similarity']) == 0:
            print("⚠️  Identity similarity data not available")
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = range(1, len(self.metrics['identity_similarity']) + 1)
        ax.plot(epochs, self.metrics['identity_similarity'], 'purple', linewidth=2.5, marker='D', markersize=5, markevery=5)
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Identity Similarity Score', fontsize=13, fontweight='bold')
        ax.set_title('Face Identity Preservation - Similarity Score', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Identity similarity plot saved to: {save_path}")
        plt.close()
    
    def plot_all_metrics(self, output_dir='outputs'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("📊 Generating metric visualizations...")
        self.plot_losses(f'{output_dir}/loss_curves.png')
        self.plot_combined_losses(f'{output_dir}/combined_losses.png')
        self.plot_identity_similarity(f'{output_dir}/identity_similarity.png')
        print(f"✅ All metrics saved to: {output_dir}")
    
    def print_summary(self):
        print("\n" + "="*60)
        print("📈 TRAINING METRICS SUMMARY")
        print("="*60)
        if 'generator_loss' in self.metrics:
            gen_losses = self.metrics['generator_loss']
            print(f"\nGenerator Loss:")
            print(f"  Initial: {gen_losses[0]:.4f}")
            print(f"  Final: {gen_losses[-1]:.4f}")
            print(f"  Min: {min(gen_losses):.4f}")
            print(f"  Max: {max(gen_losses):.4f}")
            print(f"  Mean: {np.mean(gen_losses):.4f}")
        if 'discriminator_loss' in self.metrics:
            disc_losses = self.metrics['discriminator_loss']
            print(f"\nDiscriminator Loss:")
            print(f"  Initial: {disc_losses[0]:.4f}")
            print(f"  Final: {disc_losses[-1]:.4f}")
            print(f"  Min: {min(disc_losses):.4f}")
            print(f"  Max: {max(disc_losses):.4f}")
            print(f"  Mean: {np.mean(disc_losses):.4f}")
        if 'l1_loss' in self.metrics:
            l1_losses = self.metrics['l1_loss']
            print(f"\nL1 Reconstruction Loss:")
            print(f"  Initial: {l1_losses[0]:.4f}")
            print(f"  Final: {l1_losses[-1]:.4f}")
            print(f"  Min: {min(l1_losses):.4f}")
            print(f"  Max: {max(l1_losses):.4f}")
            print(f"  Mean: {np.mean(l1_losses):.4f}")
        print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize face swap training metrics')
    parser.add_argument('--metrics-file', type=str, default='models/checkpoints/training_metrics.json', help='Path to training_metrics.json file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save plots')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')
    args = parser.parse_args()
    visualizer = MetricsVisualizer(args.metrics_file)
    visualizer.plot_all_metrics(args.output_dir)
    if args.summary:
        visualizer.print_summary()

if __name__ == '__main__':
    main()
