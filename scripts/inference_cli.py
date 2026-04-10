import argparse
import os
import sys

from face_swap_inference import FaceSwapInference


def main():
    parser = argparse.ArgumentParser(description='Run face swap inference on a single image pair')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Path to source face image')
    parser.add_argument('--target', type=str, required=True, help='Path to target face image')
    parser.add_argument('--output', type=str, required=True, help='Path to save the swapped output image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cuda or cpu')
    parser.add_argument('--img-size', type=int, default=256, help='Image size for inference')
    args = parser.parse_args()

    for label, path in [('Model', args.model), ('Source', args.source), ('Target', args.target)]:
        if not os.path.exists(path):
            print(f"❌ {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        engine = FaceSwapInference(model_path=args.model, device=args.device, img_size=args.img_size)
        swapped_image, saved_path = engine.swap_faces(
            source_path=args.source,
            target_path=args.target,
            output_path=args.output,
        )
        print(f"✅ Inference complete. Output saved to: {saved_path}")
    except Exception as e:
        print(f"❌ Inference failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
