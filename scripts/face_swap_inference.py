import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
from flask import Flask, request, jsonify, send_file
import io
from PIL import Image
import json

# ==================== IMPORT MODELS ====================
from train_faceswap_advanced import EnhancedSwapGenerator


class FaceSwapInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.generator = EnhancedSwapGenerator().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.generator.eval()

        print(f"✅ Model loaded from: {checkpoint_path}")
        print(f"📊 Epoch trained: {checkpoint['epoch']}")

        # Print metrics
        if 'metrics' in checkpoint:
            print(f"📈 Final metrics:")
            for key, values in checkpoint['metrics'].items():
                if values:
                    print(f"   {key}: {values[-1]:.4f}")

    def preprocess_image(self, image_path, img_size=256):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(self.device)

    def postprocess_image(self, tensor):
        """Convert tensor back to numpy image"""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def swap_faces(self, source_path, target_path, output_path=None):
        """Perform face swap and save if output_path is provided"""
        try:
            # Load and preprocess images
            source_tensor = self.preprocess_image(source_path)
            target_tensor = self.preprocess_image(target_path)

            print(f"🔄 Swapping faces...")
            with torch.no_grad():
                swapped = self.generator(source_tensor, target_tensor)

            # Postprocess
            swapped_image = self.postprocess_image(swapped)

            # Save if output path provided
            if output_path:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"📁 Created output directory: {output_dir}")

                # Write the image to file
                success = cv2.imwrite(output_path, swapped_image)

                if success:
                    # Verify the file actually exists and has content
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"✅ Swapped image saved: {output_path}")
                        print(f"📊 File size: {os.path.getsize(output_path) / 1024:.2f} KB")
                        return swapped_image, output_path
                    else:
                        print(f"❌ Error: File was not written correctly: {output_path}")
                        return swapped_image, None
                else:
                    print(f"❌ Error: Failed to write image to {output_path}")
                    return swapped_image, None
            else:
                print(f"✅ Face swap completed (not saved - no output path provided)")
                return swapped_image, None

        except Exception as e:
            print(f"❌ Error during face swap: {e}")
            raise

    def swap_faces_batch(self, source_path, target_dir, output_dir):
        """Swap source face with multiple target faces"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}\n")

        source_tensor = self.preprocess_image(source_path)

        results = []
        successful_count = 0
        failed_count = 0

        # Get all image files
        image_files = [f for f in os.listdir(target_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"❌ No images found in {target_dir}")
            return results

        print(f"📊 Processing {len(image_files)} target images...\n")

        for idx, target_file in enumerate(image_files, 1):
            target_path = os.path.join(target_dir, target_file)

            try:
                target_tensor = self.preprocess_image(target_path)

                with torch.no_grad():
                    swapped = self.generator(source_tensor, target_tensor)

                swapped_image = self.postprocess_image(swapped)

                # Create output filename
                name, ext = os.path.splitext(target_file)
                output_filename = f'swapped_{name}.jpg'
                output_path = os.path.join(output_dir, output_filename)

                # Write the image
                success = cv2.imwrite(output_path, swapped_image)

                if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path) / 1024
                    results.append({
                        'target': target_file,
                        'output': output_path,
                        'file_size_kb': f"{file_size:.2f}",
                        'status': 'success'
                    })
                    print(f"✅ [{idx}/{len(image_files)}] {target_file} → {output_filename} ({file_size:.2f} KB)")
                    successful_count += 1
                else:
                    results.append({
                        'target': target_file,
                        'error': 'Failed to write image file',
                        'status': 'failed'
                    })
                    print(f"❌ [{idx}/{len(image_files)}] {target_file}: Failed to write file")
                    failed_count += 1

            except Exception as e:
                results.append({
                    'target': target_file,
                    'error': str(e),
                    'status': 'failed'
                })
                print(f"❌ [{idx}/{len(image_files)}] {target_file}: {e}")
                failed_count += 1

        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"✅ Successful: {successful_count}/{len(image_files)}")
        print(f"❌ Failed: {failed_count}/{len(image_files)}")
        print(f"📁 Output directory: {output_dir}")
        print(f"{'=' * 60}\n")

        return results


# ==================== FLASK APPLICATION ====================
app = Flask(__name__)
inference_engine = None


@app.before_request
def initialize_engine():
    global inference_engine
    if inference_engine is None:
        checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'models/checkpoints/checkpoint_epoch_49.pt')
        if not os.path.exists(checkpoint_path):
            return jsonify({
                'status': 'error',
                'message': f'Checkpoint not found: {checkpoint_path}'
            }), 500
        inference_engine = FaceSwapInference(checkpoint_path)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(inference_engine.device) if inference_engine else 'not initialized'
    }), 200


@app.route('/swap', methods=['POST'])
def swap_faces():
    """Swap faces via POST with image files"""
    try:
        # Check if files are provided
        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Missing source or target image'
            }), 400

        source_file = request.files['source']
        target_file = request.files['target']

        # Save temporary files
        source_path = '/tmp/source.jpg'
        target_path = '/tmp/target.jpg'
        source_file.save(source_path)
        target_file.save(target_path)

        # Perform swap
        swapped_image, saved_path = inference_engine.swap_faces(source_path, target_path)

        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', swapped_image)
        img_io = io.BytesIO(buffer)
        img_io.seek(0)

        # Clean up
        os.remove(source_path)
        os.remove(target_path)

        return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name='swapped.jpg')

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/swap-batch', methods=['POST'])
def swap_batch():
    """Swap source face with multiple target images"""
    try:
        if 'source' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Missing source image'
            }), 400

        source_file = request.files['source']
        target_files = request.files.getlist('targets')

        if not target_files:
            return jsonify({
                'status': 'error',
                'message': 'No target images provided'
            }), 400

        # Save temporary files
        source_path = '/tmp/source.jpg'
        target_dir = '/tmp/targets'
        output_dir = '/tmp/outputs'

        source_file.save(source_path)
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        for target_file in target_files:
            target_file.save(os.path.join(target_dir, target_file.filename))

        # Perform batch swap
        results = inference_engine.swap_faces_batch(source_path, target_dir, output_dir)

        return jsonify({
            'status': 'success',
            'results': results
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'version': '2.0',
        'description': 'Advanced Face Swap Engine',
        'features': [
            'Enhanced Generator with Residual Blocks',
            'Attention Mechanisms (CBAM)',
            'Perceptual Loss (VGG19)',
            'Identity Loss Preservation',
            'Batch Processing'
        ],
        'device': str(inference_engine.device) if inference_engine else 'not initialized'
    }), 200


@app.route('/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    if inference_engine is None:
        return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500

    total_params = sum(p.numel() for p in inference_engine.generator.parameters())
    trainable_params = sum(p.numel() for p in inference_engine.generator.parameters() if p.requires_grad)

    return jsonify({
        'total_parameters': f'{total_params:,}',
        'trainable_parameters': f'{trainable_params:,}',
        'device': str(inference_engine.device)
    }), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Swap Inference Server')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/checkpoint_epoch_49.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Flask host')
    parser.add_argument('--port', type=int, default=5000, help='Flask port')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')

    args = parser.parse_args()
    os.environ['CHECKPOINT_PATH'] = args.checkpoint

    print(f"🚀 Starting Face Swap Inference Server")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"📊 Device: {args.device}\n")

    app.run(debug=False, host=args.host, port=args.port)