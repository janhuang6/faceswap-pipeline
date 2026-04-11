import argparse
import os
import sys
from face_swap_inference import FaceSwapInference


def main():
    parser = argparse.ArgumentParser(description='Face Swap Inference (CLI)')
    parser.add_argument('--source', type=str, required=True, help='Source face image path')
    parser.add_argument('--target', type=str, required=True, help='Target face image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/checkpoint_epoch_49.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.source):
        print(f"❌ Source image not found: {args.source}")
        sys.exit(1)

    if not os.path.exists(args.target):
        print(f"❌ Target image not found: {args.target}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Initialize inference engine
    print("🔄 Loading model...")
    try:
        inference = FaceSwapInference(args.checkpoint, device=args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Perform swap
    print(f"\n📸 Source: {args.source}")
    print(f"📸 Target: {args.target}")

    try:
        swapped_image, saved_path = inference.swap_faces(args.source, args.target, output_path=args.output)

        if saved_path:
            print(f"✅ Success! Swapped image saved to: {saved_path}")
            sys.exit(0)
        else:
            print(f"❌ Failed to save swapped image to: {args.output}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()