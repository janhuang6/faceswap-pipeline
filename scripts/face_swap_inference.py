import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
from flask import Flask, request, jsonify, send_file
import io
import json
import dlib

# ==================== IMPORT MODELS ====================
from train_faceswap_advanced import EnhancedSwapGenerator

# ==================== DLIB CONSTANTS ====================
PREDICTOR_PATH = "C:\\Jan\\faceswap-pipeline\\shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

COLOUR_CORRECT_BLUR_FRAC = 0.6


# ==================== EXCEPTION CLASSES ====================
class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


# ==================== LANDMARK-BASED FACE SWAP FUNCTIONS ====================
def draw_convex_hull(im, points, color):
    """Draw convex hull on image"""
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    """Generate face mask for blending"""
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im, landmarks[group], color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Calculate affine transformation between two point sets.
    Solves the procrustes problem.
    """
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    """Apply affine transformation to image"""
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    """Apply color correction for seamless blending"""
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


# ==================== FACE SWAP INFERENCE CLASS ====================
class FaceSwapInference:
    def __init__(self, checkpoint_path, device='cuda', use_landmark_blending=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.generator = EnhancedSwapGenerator().to(self.device)
        self.use_landmark_blending = use_landmark_blending

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.generator.eval()

        print("Model loaded from: " + checkpoint_path)
        print("Epoch trained: " + str(checkpoint['epoch']))

        # Initialize dlib landmark detector if blending is enabled
        if self.use_landmark_blending:
            try:
                self.detector = dlib.get_frontal_face_detector()
                self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
                print("Landmark detector initialized")
            except Exception as e:
                print("Warning: Landmark detector failed to load: " + str(e))
                print("Falling back to neural network output only")
                self.use_landmark_blending = False

        # Print metrics
        if 'metrics' in checkpoint:
            print("Final metrics:")
            for key, values in checkpoint['metrics'].items():
                if values:
                    print("   " + key + ": " + str(round(values[-1], 4)))

    def get_landmarks(self, im):
        """Detect 68 facial landmarks using dlib"""
        try:
            rects = self.detector(im, 1)

            if len(rects) > 1:
                raise TooManyFaces("Multiple faces detected")
            if len(rects) == 0:
                raise NoFaces("No faces detected")

            return np.matrix([[p.x, p.y] for p in self.predictor(im, rects[0]).parts()])
        except Exception as e:
            raise Exception("Landmark detection failed: " + str(e))

    def preprocess_image(self, image_path, img_size=256):
        """Load and preprocess image for neural network"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image: " + image_path)

        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(self.device), cv2.imread(image_path)

    def postprocess_image(self, source_img, tensor):
        """
        Convert tensor to image and apply landmark-based face swap blending
        """
        # Step 1: Convert tensor to numpy image
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Step 2: Apply landmark-based blending if enabled
        if not self.use_landmark_blending:
            return img

        try:
            # Detect landmarks in source and generated images
            landmarks_source = self.get_landmarks(source_img)
            landmarks_generated = self.get_landmarks(img)

            # Align faces using landmark-based transformation
            M = transformation_from_points(
                landmarks_source[ALIGN_POINTS],
                landmarks_generated[ALIGN_POINTS]
            )

            # Create smooth blending mask
            mask = get_face_mask(img, landmarks_generated)
            warped_mask = warp_im(mask, M, source_img.shape)

            # Warp generated image to match source face position
            warped_img = warp_im(img, M, source_img.shape)

            # Correct colors for seamless blending
            warped_corrected = correct_colours(
                source_img, warped_img, landmarks_source
            )

            # Blend using masks
            combined_mask = np.max([
                get_face_mask(source_img, landmarks_source),
                warped_mask
            ], axis=0)

            # Final output: blend source background with swapped face
            output = source_img * (1.0 - combined_mask) + warped_corrected * combined_mask
            return output.astype(np.uint8)

        except Exception as e:
            print("Warning: Landmark blending failed (" + str(e) + "), using neural network output")
            return img

    def swap_faces(self, source_path, target_path, output_path=None, save_comparison=True):
        """Perform face swap and save if output_path is provided"""
        try:
            # Load and preprocess images
            source_tensor, source_original = self.preprocess_image(source_path)
            target_tensor, target_original = self.preprocess_image(target_path)

            print("Swapping faces...")
            with torch.no_grad():
                swapped = self.generator(source_tensor, target_tensor)

            # Postprocess with landmark blending
            swapped_image = self.postprocess_image(source_original, swapped)

            # Save if output path provided
            if output_path:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    print("Created output directory: " + output_dir)

                # Write the image to file
                success = cv2.imwrite(output_path, swapped_image)

                if success:
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print("Swapped image saved: " + output_path)
                        file_size = os.path.getsize(output_path) / 1024
                        print("File size: " + str(round(file_size, 2)) + " KB")

                        # Save comparison image
                        if save_comparison:
                            comp_path = output_path.replace('.jpg', '_comparison.jpg')
                            self.save_comparison_image(
                                source_original, target_original, swapped_image, comp_path
                            )
                            print("Comparison image saved: " + comp_path)

                        return swapped_image, output_path
                    else:
                        print("Error: File was not written correctly: " + output_path)
                        return swapped_image, None
                else:
                    print("Error: Failed to write image to " + output_path)
                    return swapped_image, None
            else:
                print("Face swap completed (not saved - no output path provided)")
                return swapped_image, None

        except Exception as e:
            print("Error during face swap: " + str(e))
            raise

    def swap_faces_batch(self, source_path, target_dir, output_dir, save_comparison=True):
        """Swap source face with multiple target faces"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print("Output directory: " + output_dir + "\n")

        source_tensor, source_original = self.preprocess_image(source_path)

        results = []
        successful_count = 0
        failed_count = 0

        # Get all image files
        image_files = [f for f in os.listdir(target_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print("No images found in " + target_dir)
            return results

        print("Processing " + str(len(image_files)) + " target images...\n")

        for idx, target_file in enumerate(image_files, 1):
            target_path = os.path.join(target_dir, target_file)
            target_original = cv2.imread(target_path)

            try:
                target_tensor, _ = self.preprocess_image(target_path)

                with torch.no_grad():
                    swapped = self.generator(source_tensor, target_tensor)

                # Postprocess with landmark blending
                swapped_image = self.postprocess_image(source_original, swapped)

                # Create output filename
                name, ext = os.path.splitext(target_file)
                output_filename = 'swapped_' + name + '.jpg'
                output_path = os.path.join(output_dir, output_filename)

                # Write the image
                success = cv2.imwrite(output_path, swapped_image)

                if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    file_size = os.path.getsize(output_path) / 1024

                    # Save comparison image
                    if save_comparison:
                        comp_filename = 'comparison_' + name + '.jpg'
                        comp_path = os.path.join(output_dir, comp_filename)
                        self.save_comparison_image(
                            source_original, target_original, swapped_image, comp_path
                        )

                    results.append({
                        'target': target_file,
                        'output': output_path,
                        'file_size_kb': str(round(file_size, 2)),
                        'status': 'success'
                    })
                    print("[" + str(idx) + "/" + str(
                        len(image_files)) + "] " + target_file + " -> " + output_filename + " (" + str(
                        round(file_size, 2)) + " KB)")
                    successful_count += 1
                else:
                    results.append({
                        'target': target_file,
                        'error': 'Failed to write image file',
                        'status': 'failed'
                    })
                    print("[" + str(idx) + "/" + str(len(image_files)) + "] " + target_file + ": Failed to write file")
                    failed_count += 1

            except Exception as e:
                results.append({
                    'target': target_file,
                    'error': str(e),
                    'status': 'failed'
                })
                print("[" + str(idx) + "/" + str(len(image_files)) + "] " + target_file + ": " + str(e))
                failed_count += 1

        print("\nBATCH PROCESSING SUMMARY")
        print("=" * 60)
        print("Successful: " + str(successful_count) + "/" + str(len(image_files)))
        print("Failed: " + str(failed_count) + "/" + str(len(image_files)))
        print("Output directory: " + output_dir)
        print("=" * 60 + "\n")

        return results

    def save_comparison_image(self, source_img, target_img, swapped_img, output_path):
        """Save source, target, and swapped images side-by-side for comparison"""
        # Ensure all images are the same height
        height = 256
        width = 256

        source_img = cv2.resize(source_img, (width, height))
        target_img = cv2.resize(target_img, (width, height))
        swapped_img = cv2.resize(swapped_img, (width, height))

        # Create horizontal concatenation
        comparison = np.hstack([source_img, target_img, swapped_img])

        # Add text labels
        cv2.putText(comparison, 'Source', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Target', (256 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Swapped', (512 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save comparison image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, comparison)


# ==================== FLASK APPLICATION ====================
app = Flask(__name__)
inference_engine = None


@app.before_request
def initialize_engine():
    global inference_engine
    if inference_engine is None:
        checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'models/checkpoints/checkpoint_epoch_49.pt')
        use_blending = os.environ.get('USE_LANDMARK_BLENDING', 'true').lower() == 'true'

        if not os.path.exists(checkpoint_path):
            return jsonify({
                'status': 'error',
                'message': 'Checkpoint not found: ' + checkpoint_path
            }), 500

        inference_engine = FaceSwapInference(checkpoint_path, use_landmark_blending=use_blending)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(inference_engine.device) if inference_engine else 'not initialized',
        'landmark_blending': inference_engine.use_landmark_blending if inference_engine else False
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
        'version': '3.0',
        'description': 'Advanced Face Swap Engine with Landmark Blending',
        'features': [
            'Enhanced Generator with Residual Blocks',
            'Attention Mechanisms (CBAM)',
            'Perceptual Loss (VGG19)',
            'Identity Loss Preservation',
            'Landmark-based Face Alignment and Blending',
            'Color Correction for Seamless Integration',
            'Batch Processing'
        ],
        'device': str(inference_engine.device) if inference_engine else 'not initialized',
        'landmark_blending': inference_engine.use_landmark_blending if inference_engine else False
    }), 200


@app.route('/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    if inference_engine is None:
        return jsonify({'status': 'error', 'message': 'Engine not initialized'}), 500

    total_params = sum(p.numel() for p in inference_engine.generator.parameters())
    trainable_params = sum(p.numel() for p in inference_engine.generator.parameters() if p.requires_grad)

    return jsonify({
        'total_parameters': '{:,}'.format(total_params),
        'trainable_parameters': '{:,}'.format(trainable_params),
        'device': str(inference_engine.device),
        'landmark_blending_enabled': inference_engine.use_landmark_blending
    }), 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Swap Inference Server with Landmark Blending')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/checkpoint_epoch_49.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Flask host')
    parser.add_argument('--port', type=int, default=5000, help='Flask port')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--no-landmark-blending', action='store_true',
                        help='Disable landmark-based blending (use NN output only)')

    args = parser.parse_args()
    os.environ['CHECKPOINT_PATH'] = args.checkpoint
    os.environ['USE_LANDMARK_BLENDING'] = 'false' if args.no_landmark_blending else 'true'

    print("Starting Face Swap Inference Server")
    print("Host: " + args.host)
    print("Port: " + str(args.port))
    print("Device: " + args.device)
    print("Landmark Blending: " + ("Disabled" if args.no_landmark_blending else "Enabled") + "\n")

    app.run(debug=False, host=args.host, port=args.port)