#!/usr/bin/env python3
"""
Run InternVideo2.5-Chat-8B inference on RoadBuddy public test dataset.

Usage:
    # With 8-bit quantization (for 24GB VRAM)
    python run_inference.py --load_in_8bit

    # Without quantization (requires ~40GB VRAM)
    python run_inference.py

    # Custom paths
    python run_inference.py --input_json /path/to/test.json --data_dir /path/to/videos --output_dir /path/to/output

    # Adjust frame sampling
    python run_inference.py --load_in_8bit --num_segments 64  # Faster, less accurate
    python run_inference.py --load_in_8bit --num_segments 256  # Slower, more accurate

    # Test with limited samples
    python run_inference.py --load_in_8bit --samples 1  # Test with 1 sample
    python run_inference.py --load_in_8bit --samples 10  # Test with 10 samples
"""

import argparse
from datetime import datetime
from pathlib import Path

from inference_internvideo_2_5_8B import InternVideo2Inference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run InternVideo2.5-Chat-8B inference on RoadBuddy dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input_json',
        type=str,
        default='../../RoadBuddy/traffic_buddy_train+public_test/public_test/public_test.json',
        help='Path to test JSON file (default: RoadBuddy public_test.json)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../../RoadBuddy/traffic_buddy_train+public_test/',
        help='Base directory containing videos (default: RoadBuddy data directory)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='../output/',
        help='Output directory for CSV results (default: ../output/)'
    )

    parser.add_argument(
        '--load_in_8bit',
        action='store_true',
        help='Enable 8-bit quantization to reduce VRAM usage (recommended for 24GB VRAM)'
    )

    parser.add_argument(
        '--num_segments',
        type=int,
        default=128,
        help='Number of frames to sample from each video (default: 128). Higher = more accurate but slower.'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='OpenGVLab/InternVideo2_5_Chat_8B',
        help='HuggingFace model identifier or local path (default: OpenGVLab/InternVideo2_5_Chat_8B)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to process (default: None = all samples). Useful for testing.'
    )

    return parser.parse_args()


def main():
    """Main entry point for inference."""
    args = parse_args()

    # Validate input paths
    input_json_path = Path(args.input_json)
    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "InternVideo2_5_8B"
    output_csv_path = output_dir / f"public_test_{model_name}_{timestamp}.csv"

    # Print configuration
    print("=" * 80)
    print("InternVideo2.5-Chat-8B Inference Configuration")
    print("=" * 80)
    print(f"Model path:         {args.model_path}")
    print(f"8-bit quantization: {'Enabled' if args.load_in_8bit else 'Disabled'}")
    print(f"Frame segments:     {args.num_segments}")
    print(f"Sample limit:       {args.samples if args.samples else 'All samples'}")
    print(f"Input JSON:         {input_json_path}")
    print(f"Data directory:     {data_dir}")
    print(f"Output CSV:         {output_csv_path}")
    print("=" * 80)
    print()

    # Initialize model
    print("Initializing model...")
    inferencer = InternVideo2Inference(
        model_path=args.model_path,
        load_in_8bit=args.load_in_8bit
    )
    print()

    # Run inference pipeline
    print("Starting inference pipeline...")
    inferencer.run_pipeline(
        test_json_path=str(input_json_path),
        data_dir=str(data_dir),
        output_csv_path=str(output_csv_path),
        num_segments=args.num_segments,
        max_samples=args.samples
    )

    print()
    print("=" * 80)
    print("Inference completed successfully!")
    print(f"Results saved to: {output_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
