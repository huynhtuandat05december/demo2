import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class InternVideo2Inference:
    """InternVideo2.5-Chat-8B inference pipeline for video question answering."""

    def __init__(self, model_path: str = 'OpenGVLab/InternVideo2_5_Chat_8B', load_in_8bit: bool = False):
        """
        Initialize the InternVideo2.5 model.

        Args:
            model_path: HuggingFace model identifier or local path
            load_in_8bit: Whether to load model in 8-bit quantization (reduces VRAM usage)
        """
        print(f"Loading model: {model_path}")
        print(f"8-bit quantization: {'Enabled' if load_in_8bit else 'Disabled'}")

        self.model_path = model_path
        self.load_in_8bit = load_in_8bit

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Load model with optional 8-bit quantization
        if load_in_8bit:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True
            ).half().cuda().to(torch.bfloat16)

        # Image preprocessing constants
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        print(f"Model loaded successfully on {self.model.device}")

    def build_transform(self, input_size: int = 448):
        """Build image transformation pipeline."""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: List[Tuple[int, int]],
        width: int,
        height: int,
        image_size: int
    ) -> Tuple[int, int]:
        """Find the closest aspect ratio from target ratios."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 6,
        image_size: int = 448,
        use_thumbnail: bool = False
    ) -> List[Image.Image]:
        """Dynamically preprocess image based on aspect ratio."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        assert len(processed_images) == blocks

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def load_image(self, image: Image.Image, input_size: int = 448, max_num: int = 6) -> torch.Tensor:
        """Load and preprocess a single image."""
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_index(
        self,
        bound: Optional[Tuple[float, float]],
        fps: float,
        max_frame: int,
        first_idx: int = 0,
        num_segments: int = 32
    ) -> np.ndarray:
        """Get frame indices for video sampling."""
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000

        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments

        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])

        return frame_indices

    def get_num_frames_by_duration(self, duration: float) -> int:
        """Calculate optimal number of frames based on video duration."""
        local_num_frames = 4
        num_segments = int(duration // local_num_frames)

        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        # Clamp between 128 and 512
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)

        return num_frames

    def load_video(
        self,
        video_path: str,
        bound: Optional[Tuple[float, float]] = None,
        input_size: int = 448,
        max_num: int = 1,
        num_segments: int = 32,
        get_frame_by_duration: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        """Load and preprocess video frames."""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)

        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = self.get_num_frames_by_duration(duration)

        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def build_prompt(self, question: str, choices: List[str]) -> str:
        """
        Build Vietnamese prompt for dashcam video question answering.

        Args:
            question: The question to ask about the video
            choices: List of answer choices (e.g., ['A. ...', 'B. ...', ...])

        Returns:
            Formatted prompt string
        """
        choices_text = "\n".join(choices)

        prompt = f"""Đây là video từ camera hành trình của xe ô tô đang di chuyển trên đường. Hãy phân tích video và trả lời câu hỏi sau:

Câu hỏi: {question}

Các lựa chọn:
{choices_text}

Hãy chọn đáp án đúng nhất và chỉ trả lời bằng chữ cái A, B, C, hoặc D."""

        return prompt

    def parse_answer(self, output: str) -> str:
        """
        Parse answer from model output.

        Args:
            output: Raw model output text

        Returns:
            Single letter answer (A, B, C, or D), defaults to 'A' if parsing fails
        """
        # Try to find answer patterns
        patterns = [
            r'\b([A-D])\b',  # Single letter
            r'[Đđ]áp án:?\s*([A-D])',  # "Đáp án: A"
            r'[Cc]họn:?\s*([A-D])',  # "Chọn: A"
            r'[Tt]rả lời:?\s*([A-D])',  # "Trả lời: A"
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Default to A if no match found
        print(f"Warning: Could not parse answer from output: {output[:100]}...")
        return 'A'

    def infer(
        self,
        video_path: str,
        question: str,
        choices: List[str],
        num_segments: int = 128
    ) -> str:
        """
        Run inference on a single video.

        Args:
            video_path: Path to video file
            question: Question to ask about the video
            choices: List of answer choices
            num_segments: Number of frames to sample from video

        Returns:
            Parsed answer (A, B, C, or D)
        """
        # Generation config
        generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1
        )

        with torch.no_grad():
            # Load video
            pixel_values, num_patches_list = self.load_video(
                video_path,
                num_segments=num_segments,
                max_num=1,
                get_frame_by_duration=False
            )

            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)

            # Build prompt
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            prompt = self.build_prompt(question, choices)
            full_prompt = video_prefix + prompt

            # Get model response
            output, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )

            # Parse answer
            answer = self.parse_answer(output)

            return answer

    def infer_batch(
        self,
        test_data: List[Dict],
        data_dir: Path,
        num_segments: int = 128
    ) -> List[Dict[str, str]]:
        """
        Run batch inference on multiple videos.

        Args:
            test_data: List of test samples with 'id', 'question', 'choices', 'video_path'
            data_dir: Base directory containing videos
            num_segments: Number of frames to sample from each video

        Returns:
            List of dictionaries with 'id' and 'answer' keys
        """
        results = []

        for item in tqdm(test_data, desc="Processing videos"):
            try:
                # Construct full video path
                video_path = data_dir / item['video_path']

                if not video_path.exists():
                    print(f"Warning: Video not found: {video_path}")
                    results.append({'id': item['id'], 'answer': 'A'})
                    continue

                # Run inference
                answer = self.infer(
                    str(video_path),
                    item['question'],
                    item['choices'],
                    num_segments=num_segments
                )

                results.append({'id': item['id'], 'answer': answer})

                # Clear CUDA cache to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {item['id']}: {str(e)}")
                results.append({'id': item['id'], 'answer': 'A'})

        return results

    def run_pipeline(
        self,
        test_json_path: str,
        data_dir: str,
        output_csv_path: str,
        num_segments: int = 128,
        max_samples: Optional[int] = None
    ):
        """
        Complete inference pipeline: load data -> infer -> save CSV.

        Args:
            test_json_path: Path to test JSON file
            data_dir: Base directory containing videos
            output_csv_path: Output CSV file path
            num_segments: Number of frames to sample from each video
            max_samples: Maximum number of samples to process (None = all samples)
        """
        # Load test data
        print(f"Loading test data from: {test_json_path}")
        with open(test_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_data = data['data']
        print(f"Loaded {len(test_data)} test samples")

        # Limit samples if requested
        if max_samples is not None:
            test_data = test_data[:max_samples]
            print(f"Processing limited to {len(test_data)} samples")

        # Run batch inference
        results = self.infer_batch(test_data, Path(data_dir), num_segments)

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"\nResults saved to: {output_csv_path}")
        print(f"Total samples: {len(results)}")


# Example usage
if __name__ == "__main__":
    # Initialize model
    model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
    load_in_8bit = False  # Set to True for 24GB VRAM

    inferencer = InternVideo2Inference(model_path=model_path, load_in_8bit=load_in_8bit)

    # Run pipeline
    test_json_path = "../RoadBuddy/traffic_buddy_train+public_test/public_test/public_test.json"
    data_dir = "../RoadBuddy/traffic_buddy_train+public_test/"

    # Generate output filename with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = f"../output/public_test_InternVideo2_5_8B_{timestamp}.csv"

    inferencer.run_pipeline(
        test_json_path=test_json_path,
        data_dir=data_dir,
        output_csv_path=output_csv_path,
        num_segments=128
    )
