import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np


class FaceSwapMetricsAnalyzer:
    def __init__(self, metrics_file):
        self.metrics_file = Path(metrics_file)
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        with open(self.metrics_file, 'r') as f:
            self.metrics = json.load(f)

        sns.set_style("whitegrid")
        self.colors = {
            'generator': '#1f77b4',
            'discriminator': '#ff7f0e',
            'l1': '#2ca02c',
            'perceptual': '#d62728',
            'identity': '#9467bd',
            'adversarial': '#8c564b'
        }

    def plot_all_losses(self, save_path='results/all_losses.png'):
        """Plot all loss components"""
        fig, ax = plt.subplots(figsize=(14, 7))
        epochs = range(1, len(self.metrics['generator_loss']) + 1)

        ax.plot(epochs, self.metrics['generator_loss'],
                color=self.colors['generator'], linewidth=2.5, label='Generator Loss', marker='o', markersize=3,
                markevery=max(1, len(epochs) // 10))
        ax.plot(epochs, self.metrics['discriminator_loss'],
                color=self.colors['discriminator'], linewidth=2.5, label='Discriminator Loss', marker='s', markersize=3,
                markevery=max(1, len(epochs) // 10))
        ax.plot(epochs, self.metrics['l1_loss'],
                color=self.colors['l1'], linewidth=2.5, label='L1 Loss', marker='^', markersize=3,
                markevery=max(1, len(epochs) // 10))

        if 'perceptual_loss' in self.metrics and self.metrics['perceptual_loss']:
            ax.plot(epochs, self.metrics['perceptual_loss'],
                    color=self.colors['perceptual'], linewidth=2.5, label='Perceptual Loss', marker='d', markersize=3,
                    markevery=max(1, len(epochs) // 10))

        if 'identity_loss' in self.metrics and self.metrics['identity_loss']:
            ax.plot(epochs, self.metrics['identity_loss'],
                    color=self.colors['identity'], linewidth=2.5, label='Identity Loss', marker='v', markersize=3,
                    markevery=max(1, len(epochs) // 10))

        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=13, fontweight='bold')
        ax.set_title('Face Swap Training - All Loss Components', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All losses plot saved to: {save_path}")
        plt.close()

    def plot_generator_components(self, save_path='results/generator_components.png'):
        """Plot what makes up the generator loss"""
        fig, ax = plt.subplots(figsize=(14, 7))
        epochs = range(1, len(self.metrics['generator_loss']) + 1)

        ax.plot(epochs, self.metrics['l1_loss'],
                color=self.colors['l1'], linewidth=2.5, label='L1 Loss (Reconstruction)', marker='^', markersize=3,
                markevery=max(1, len(epochs) // 10))

        if 'perceptual_loss' in self.metrics and self.metrics['perceptual_loss']:
            ax.plot(epochs, self.metrics['perceptual_loss'],
                    color=self.colors['perceptual'], linewidth=2.5, label='Perceptual Loss (VGG)', marker='d',
                    markersize=3, markevery=max(1, len(epochs) // 10))

        if 'identity_loss' in self.metrics and self.metrics['identity_loss']:
            ax.plot(epochs, self.metrics['identity_loss'],
                    color=self.colors['identity'], linewidth=2.5, label='Identity Loss (Source Preservation)',
                    marker='v', markersize=3, markevery=max(1, len(epochs) // 10))

        if 'adversarial_loss' in self.metrics and self.metrics['adversarial_loss']:
            ax.plot(epochs, self.metrics['adversarial_loss'],
                    color=self.colors['adversarial'], linewidth=2.5, label='Adversarial Loss', marker='s', markersize=3,
                    markevery=max(1, len(epochs) // 10))

        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=13, fontweight='bold')
        ax.set_title('Face Swap Training - Generator Loss Components', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Generator components plot saved to: {save_path}")
        plt.close()

    def plot_gan_balance(self, save_path='results/gan_balance.png'):
        """Plot generator vs discriminator balance (critical for GAN training)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(self.metrics['generator_loss']) + 1)

        # Plot 1: G vs D loss
        ax1.plot(epochs, self.metrics['generator_loss'],
                 color=self.colors['generator'], linewidth=2.5, label='Generator Loss', marker='o', markersize=3,
                 markevery=max(1, len(epochs) // 10))
        ax1.plot(epochs, self.metrics['discriminator_loss'],
                 color=self.colors['discriminator'], linewidth=2.5, label='Discriminator Loss', marker='s',
                 markersize=3, markevery=max(1, len(epochs) // 10))
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax1.set_title('Generator vs Discriminator Loss (Balance)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # Plot 2: Ratio (diagnostic)
        g_loss = np.array(self.metrics['generator_loss'])
        d_loss = np.array(self.metrics['discriminator_loss'])

        # Avoid division by zero
        ratio = np.where(d_loss > 0.01, g_loss / d_loss, 1.0)

        ax2.plot(epochs, ratio, color='purple', linewidth=2.5, marker='o', markersize=3,
                 markevery=max(1, len(epochs) // 10))
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal Ratio (1:1)')
        ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=2, label='Good Range (0.5-2.0)')
        ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('G Loss / D Loss Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('GAN Balance Metric (Healthy: 0.5-2.0)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"GAN balance plot saved to: {save_path}")
        plt.close()

    def plot_convergence_analysis(self, save_path='results/convergence_analysis.png'):
        """Analyze training convergence trends"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.metrics['generator_loss']) + 1)

        # Plot 1: Generator loss trend
        ax = axes[0, 0]
        g_loss = self.metrics['generator_loss']
        ax.plot(epochs, g_loss, color=self.colors['generator'], linewidth=2.5, marker='o', markersize=3,
                markevery=max(1, len(epochs) // 10))
        # Add trend line
        z = np.polyfit(epochs, g_loss, 2)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.set_title('Generator Loss Convergence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Discriminator loss trend
        ax = axes[0, 1]
        d_loss = self.metrics['discriminator_loss']
        ax.plot(epochs, d_loss, color=self.colors['discriminator'], linewidth=2.5, marker='s', markersize=3,
                markevery=max(1, len(epochs) // 10))
        z = np.polyfit(epochs, d_loss, 2)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.set_title('Discriminator Loss Convergence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 3: Identity loss trend (most important for face swap!)
        ax = axes[1, 0]
        if 'identity_loss' in self.metrics and self.metrics['identity_loss']:
            id_loss = self.metrics['identity_loss']
            ax.plot(epochs, id_loss, color=self.colors['identity'], linewidth=2.5, marker='v', markersize=3,
                    markevery=max(1, len(epochs) // 10))
            z = np.polyfit(epochs, id_loss, 2)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax.set_title('Identity Loss Convergence (Source Preservation)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Loss', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        stats_text = f"""
TRAINING STATISTICS
{'=' * 50}

Generator Loss:
  • Initial: {g_loss[0]:.4f}
  • Final: {g_loss[-1]:.4f}
  • Change: {g_loss[-1] - g_loss[0]:.4f}
  • Trend: {'Decreasing' if g_loss[-1] < g_loss[0] else 'Increasing'}

Discriminator Loss:
  • Initial: {d_loss[0]:.4f}
  • Final: {d_loss[-1]:.4f}
  • Change: {d_loss[-1] - d_loss[0]:.4f}

Identity Loss:
  • Initial: {self.metrics['identity_loss'][0]:.4f}
  • Final: {self.metrics['identity_loss'][-1]:.4f}
  • Trend: {'Better (Source preserved)' if self.metrics['identity_loss'][-1] < self.metrics['identity_loss'][0] else '↑ Worse'}

Total Epochs: {len(epochs)}

INTERPRETATION:
G loss decreasing = Better generation
D loss stable = Balanced training
Identity loss decreasing = Better source preservation
        """

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis plot saved to: {save_path}")
        plt.close()

    def generate_all_plots(self, output_dir='results'):
        """Generate all visualization plots"""
        print(f"\nGenerating training analysis plots...\n")

        self.plot_all_losses(f'{output_dir}/01_all_losses.png')
        self.plot_generator_components(f'{output_dir}/02_generator_components.png')
        self.plot_gan_balance(f'{output_dir}/03_gan_balance.png')
        self.plot_convergence_analysis(f'{output_dir}/04_convergence_analysis.png')

        print(f"\nAll plots saved to: {output_dir}/")
        print(f"Generated files:")
        print(f"  1. 01_all_losses.png - All loss components")
        print(f"  2. 02_generator_components.png - Generator loss breakdown")
        print(f"  3. 03_gan_balance.png - G vs D balance analysis")
        print(f"  4. 04_convergence_analysis.png - Convergence trends\n")


def main():
    parser = argparse.ArgumentParser(description='Advanced Face Swap Training Analysis')
    parser.add_argument('--metrics', type=str,
                        default='models/checkpoints/training_metrics.json',
                        help='Path to training metrics JSON file')
    parser.add_argument('--output', type=str,
                        default='results',
                        help='Output directory for plots')

    args = parser.parse_args()

    try:
        analyzer = FaceSwapMetricsAnalyzer(args.metrics)
        analyzer.generate_all_plots(args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure training_metrics.json exists at: {args.metrics}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()