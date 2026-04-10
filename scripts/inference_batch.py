import argparse
import os
import sys
from pathlib import Path

from face_swap_inference import FaceSwapInference


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def collect_images(directory):
    paths = []
    for p in Path(directory).rglob('*'):
        if p.suffix.lower() in VALID_EXTENSIONS:
            paths.append(str(p))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description='Run face swap inference in batch mode')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--source-dir', type=str, required=True, help='Directory containing source face images')
    parser.add_argument('--target-dir', type=str, required=True, help='Directory containing target face images')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save swapped output images')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cuda or cpu')
    parser.add_argument('--img-size', type=int, default=256, help='Image size for inference')
    args = parser.parse_args()

    for label, path in [('Model', args.model), ('Source dir', args.source_dir), ('Target dir', args.target_dir)]:
        if not os.path.exists(path):
            print(f"❌ {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    source_images = collect_images(args.source_dir)
    target_images = collect_images(args.target_dir)

    if not source_images:
        print(f"❌ No images found in source directory: {args.source_dir}", file=sys.stderr)
        sys.exit(1)
    if not target_images:
        print(f"❌ No images found in target directory: {args.target_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        engine = FaceSwapInference(model_path=args.model, device=args.device, img_size=args.img_size)
    except Exception as e:
        print(f"❌ Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    succeeded = 0
    failed = 0
    total = len(target_images)

    for i, target_path in enumerate(target_images):
        source_path = source_images[i % len(source_images)]  # Cycle through source images if fewer sources than targets
        target_name = Path(target_path).stem
        output_path = os.path.join(args.output_dir, f"{target_name}_swapped.jpg")

        try:
            _, saved_path = engine.swap_faces(
                source_path=source_path,
                target_path=target_path,
                output_path=output_path,
            )
            print(f"[{i + 1}/{total}] ✅ Saved: {saved_path}")
            succeeded += 1
        except Exception as e:
            print(f"[{i + 1}/{total}] ❌ Failed ({Path(target_path).name}): {e}", file=sys.stderr)
            failed += 1

    print(f"\n📊 Batch complete — succeeded: {succeeded}, failed: {failed}, total: {total}")
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
