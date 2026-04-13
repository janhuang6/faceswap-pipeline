import cv2
import numpy as np
from pathlib import Path


def analyze_comparison_image(comp_image_path):
    """Analyze the comparison image to diagnose issues"""
    img = cv2.imread(comp_image_path)
    if img is None:
        print(f"Image not found: {comp_image_path}")
        return

    # Split into 3 parts
    height, width = img.shape[:2]
    third = width // 3

    source = img[:, :third]
    target = img[:, third:2 * third]
    swapped = img[:, 2 * third:]

    # Check sharpness using Laplacian variance
    def get_sharpness(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    # Check brightness
    def get_brightness(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    # Check if swapped has source facial features
    def compute_difference(img1, img2):
        """How different are two images"""
        diff = cv2.absdiff(img1, img2)
        return np.mean(diff)

    source_sharp = get_sharpness(source)
    target_sharp = get_sharpness(target)
    swapped_sharp = get_sharpness(swapped)

    source_bright = get_brightness(source)
    target_bright = get_brightness(target)
    swapped_bright = get_brightness(swapped)

    source_vs_swapped = compute_difference(source, swapped)
    target_vs_swapped = compute_difference(target, swapped)

    print(f"\nPIPELINE DIAGNOSTIC ANALYSIS")
    print(f"{'=' * 60}")

    print(f"\nSharpness (higher = sharper):")
    print(f"  Source: {source_sharp:.2f}")
    print(f"  Target: {target_sharp:.2f}")
    print(f"  Swapped: {swapped_sharp:.2f}")

    if swapped_sharp < 50:
        print(f"WARNING: Swapped image is BLURRY!")
        print(f"  Reduce L1 loss or increase perceptual loss")
    elif swapped_sharp > target_sharp * 0.8:
        print(f"  Swapped sharpness is good")

    print(f"\nBrightness Consistency:")
    print(f"  Source: {source_bright:.2f}")
    print(f"  Target: {target_bright:.2f}")
    print(f"  Swapped: {swapped_bright:.2f}")

    if abs(swapped_bright - target_bright) < 20:
        print(f"  Swapped matches target brightness")
    else:
        print(f"  Swapped brightness differs from target")

    print(f"\nFace Identity Preservation:")
    print(f"  Difference from Source: {source_vs_swapped:.2f}")
    print(f"  Difference from Target: {target_vs_swapped:.2f}")

    # Lower difference = more similar
    if source_vs_swapped < target_vs_swapped * 1.5:
        print(f"  Swapped preserves source features well")
    else:
        print(f"  Swapped is ignoring source features!")
        print(f"      Increase identity loss weight")

    if target_vs_swapped < 30:
        print(f"  Swapped is too close to target (just copying)")
        print(f"      Increase identity loss, reduce L1 loss")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagnose_pipeline.py <comparison_image_path>")
        sys.exit(1)

    analyze_comparison_image(sys.argv[1])