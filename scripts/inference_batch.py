import argparse
import os
import sys
from face_swap_inference import FaceSwapInference


def main():
    parser = argparse.ArgumentParser(description='Face Swap Batch Inference')
    parser.add_argument('--source', type=str, required=True, help='Source face image path')
    parser.add_argument('--target-dir', type=str, required=True, help='Directory with target images')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for swapped images')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/checkpoint_epoch_49.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.source):
        print(f"❌ Source image not found: {args.source}")
        sys.exit(1)

    if not os.path.isdir(args.target_dir):
        print(f"❌ Target directory not found: {args.target_dir}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Initialize inference engine
    print("🔄 Loading model...\n")
    try:
        inference = FaceSwapInference(args.checkpoint, device=args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Perform batch swap
    print(f"\n�� Source: {args.source}")
    print(f"📁 Target directory: {args.target_dir}\n")

    try:
        results = inference.swap_faces_batch(args.source, args.target_dir, args.output_dir)

        # Count results
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')

        if successful > 0:
            print(f"✅ All files saved successfully!")
            # List saved files
            print(f"\n📁 Saved files in {args.output_dir}:")
            for result in results:
                if result['status'] == 'success':
                    print(f"   ✅ {os.path.basename(result['output'])} ({result['file_size_kb']} KB)")
            sys.exit(0)
        else:
            print(f"❌ All batch processing failed!")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during batch inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()