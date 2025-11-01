import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm
import gc


def log_vram_usage(stage=""):
    """Log current VRAM usage for monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {stage}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
    return allocated if torch.cuda.is_available() else 0


def load_grounding_dino(model_id="IDEA-Research/grounding-dino-tiny", device=None):
    """Load Grounding DINO model for object detection"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Grounding DINO model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    print("Grounding DINO loaded successfully!")

    return processor, model


def preprocess_frame_with_grounding_dino(frame, processor, model, threshold=0.3, text_threshold=0.25):
    """
    Detect objects in frame using Grounding DINO and return cropped regions

    Args:
        frame: PIL Image
        processor: Grounding DINO processor
        model: Grounding DINO model
        threshold: Detection confidence threshold
        text_threshold: Text matching threshold

    Returns:
        list of (PIL.Image, label, score): Cropped regions with their labels and scores
    """
    # Define traffic-related object categories
    text_labels = [[
        "traffic sign", "traffic light", "traffic signal",
        "lane arrow", "road marking", "lane marking",
        "vehicle", "car", "truck", "motorcycle", "bicycle",
        "pedestrian", "stop sign", "speed limit sign"
    ]]

    # Run detection
    inputs = processor(images=frame, text=text_labels, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[frame.size[::-1]]  # (height, width)
    )

    result = results[0]
    cropped_regions = []

    # Extract and crop detected regions
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        box = box.tolist()
        # Convert box coordinates to integers: [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = map(int, box)

        # Crop the region from the frame
        cropped = frame.crop((x_min, y_min, x_max, y_max))
        cropped_regions.append((cropped, label, score.item()))

    return cropped_regions


def create_prompt(question, choices, video_prefix, detections_info=None):
    """Create prompt with optional detection information"""
    detection_context = ""
    if detections_info and any(info['num_detections'] > 0 for info in detections_info):
        detection_context = "\n    Các đối tượng được phát hiện trong video:"
        for i, info in enumerate(detections_info):
            if info['num_detections'] > 0:
                labels_str = ", ".join(info['labels'])
                detection_context += f"\n    - Frame {i+1}: {info['num_detections']} đối tượng ({labels_str})"

    SYSTEM_PROMPT = f"""
    Bạn là một AI chuyên gia phân tích an toàn giao thông. Nhiệm vụ duy nhất của bạn là phân tích video clip từ camera hành trình được cung cấp và trả lời một câu hỏi cụ thể về video đó.

    Nguyên tắc phân tích:

    Chỉ dựa vào hình ảnh: Câu trả lời của bạn chỉ được dựa trên những gì xuất hiện trực quan trong các khung hình của video.

    Tập trung vào đối tượng: Chú ý kỹ đến đèn giao thông, biển báo (giới hạn tốc độ, dừng, cảnh báo), vạch kẻ đường, các phương tiện khác (ô tô, xe tải, xe máy), người đi bộ, và điều kiện thời tiết/đường sá.

    Nhận thức về thời gian: Xem xét chuỗi sự kiện. Nếu câu hỏi về một hành động, hãy mô tả những gì xảy ra trong suốt clip.

    Tuân thủ định dạng: Đối với các câu hỏi trắc nghiệm, chỉ trả lời bằng chữ cái (ví dụ: A, B, C, D) của lựa chọn đúng. Không giải thích.
    {detection_context}

    {video_prefix}

    Câu hỏi: {question}

    Các lựa chọn: {choices}

    Chỉ trả lời bằng chữ cái (A, B, C, hoặc D) tương ứng với lựa chọn đúng.
    """
    return SYSTEM_PROMPT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=8):
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

def load_video(video_path, bound=None, input_size=448, max_num=3, num_segments=8,
               grounding_dino_processor=None, grounding_dino_model=None,
               detection_threshold=0.3, max_detections_per_frame=5):
    """Load video and extract frames uniformly with optional Grounding DINO preprocessing"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    detections_info = []  # Track detection info for each frame
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    use_grounding_dino = grounding_dino_processor is not None and grounding_dino_model is not None

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')

        frame_detection_info = {'num_detections': 0, 'labels': []}

        if use_grounding_dino:
            # Detect objects in the frame
            cropped_regions = preprocess_frame_with_grounding_dino(
                img, grounding_dino_processor, grounding_dino_model,
                threshold=detection_threshold
            )

            # Limit number of detections to avoid too many patches
            cropped_regions = cropped_regions[:max_detections_per_frame]
            frame_detection_info['num_detections'] = len(cropped_regions)
            frame_detection_info['labels'] = [label for _, label, _ in cropped_regions]

            # Process original frame (disable thumbnail to reduce patches)
            img_patches = dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)

            # Process cropped regions and add them
            for cropped_img, label, score in cropped_regions:
                # Process each detected region
                cropped_patches = dynamic_preprocess(cropped_img, image_size=input_size,
                                                     use_thumbnail=False, max_num=1)
                img_patches.extend(cropped_patches)
        else:
            # Original behavior without Grounding DINO
            img_patches = dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)

        # Transform all patches
        pixel_values = [transform(tile) for tile in img_patches]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
        detections_info.append(frame_detection_info)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list, detections_info

def split_model(model_path):
    """Split model across multiple GPUs"""
    device_map = {}
    world_size = torch.cuda.device_count()

    if world_size == 1:
        return 'auto'

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def load_model(model_path, load_in_8bit=True):
    """Load InternVL3 model with optional 8-bit quantization"""
    print(f"Loading model: {model_path}")
    print(f"8-bit quantization: {'Enabled' if load_in_8bit else 'Disabled'}")

    device_map = split_model(model_path)

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    print("Model loaded successfully!")
    return model, tokenizer

def load_test_data(json_path, samples=None):
    """Load public test data from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data['data']

    if samples is not None:
        questions = questions[:samples]
        print(f"Processing {samples} samples (out of {data['__count__']} total)")
    else:
        print(f"Processing all {len(questions)} questions")

    return questions

# def create_prompt(question, choices):
#     """Create prompt for Vietnamese traffic video Q&A"""
#     choices_text = '\n'.join(choices)

#     prompt = f"""Bạn là một trợ lý AI chuyên về giao thông. Dựa trên video giao thông được cung cấp, hãy trả lời câu hỏi sau.

# Câu hỏi: {question}

# Các lựa chọn:
# {choices_text}

# Hãy chọn đáp án đúng nhất (chỉ trả lời A, B, C hoặc D):"""

#     return prompt

def extract_answer(response, num_choices=4):
    """Extract answer letter from model response"""
    # Try to find A, B, C, or D in the response
    response_upper = response.upper().strip()

    # Check if response starts with a letter
    if response_upper and response_upper[0] in ['A', 'B', 'C', 'D']:
        answer = response_upper[0]
        # Validate against number of choices
        valid_choices = ['A', 'B', 'C', 'D'][:num_choices]
        if answer in valid_choices:
            return answer

    # Search for patterns like "A.", "A)", "A:", etc.
    import re
    patterns = [
        r'\b([A-D])\.',
        r'\b([A-D])\)',
        r'\b([A-D]):',
        r'^([A-D])\b',
        r'\b([A-D])\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            answer = match.group(1)
            valid_choices = ['A', 'B', 'C', 'D'][:num_choices]
            if answer in valid_choices:
                return answer

    # Default fallback
    print(f"Warning: Could not extract answer from: {response[:100]}... Using 'A' as fallback.")
    return 'A'

def run_inference(model, tokenizer, questions, base_path, num_frames=8,
                  grounding_dino_processor=None, grounding_dino_model=None,
                  detection_threshold=0.3, max_detections_per_frame=5, max_num=3):
    """Run inference on all questions"""
    results = []
    video_cache = {}  # Cache video frames

    generation_config = dict(max_new_tokens=512, do_sample=False, temperature=0.0)

    # Log VRAM at start of inference
    log_vram_usage("Start of inference")

    for idx, item in enumerate(tqdm(questions, desc="Processing questions")):
        question_id = item['id']
        question_text = item['question']
        choices = item['choices']
        video_path = item['video_path']

        # Construct full video path
        full_video_path = os.path.join(base_path, video_path)

        # Check if video has been processed before (caching)
        if video_path not in video_cache:
            try:
                pixel_values, num_patches_list, detections_info = load_video(
                    full_video_path,
                    num_segments=num_frames,
                    max_num=max_num,
                    grounding_dino_processor=grounding_dino_processor,
                    grounding_dino_model=grounding_dino_model,
                    detection_threshold=detection_threshold,
                    max_detections_per_frame=max_detections_per_frame
                )
                # Store on CPU to save VRAM - will move to GPU only during inference
                pixel_values = pixel_values.to(torch.bfloat16).cpu()
                video_cache[video_path] = (pixel_values, num_patches_list, detections_info)
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                results.append({'id': question_id, 'answer': 'A', 'raw_response': f'Error: {str(e)}'})
                continue
        else:
            pixel_values, num_patches_list, detections_info = video_cache[video_path]

        # Create video prefix with frame markers
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

        # Create full prompt with detection info
        prompt = create_prompt(question_text, choices, video_prefix, detections_info)

        try:
            # Move pixel_values to GPU for inference (from CPU cache)
            pixel_values_gpu = pixel_values.cuda()

            # Run inference with torch.inference_mode() for memory efficiency
            with torch.inference_mode():
                response = model.chat(
                    tokenizer,
                    pixel_values_gpu,
                    prompt,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )

            # Extract answer
            answer = extract_answer(response, len(choices))

            results.append({
                'id': question_id,
                'answer': answer,
                'raw_response': response,
                'prompt': prompt
            })

            # Free GPU memory immediately after inference
            del pixel_values_gpu
            torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection

            # Log VRAM periodically (every 10 questions)
            if (idx + 1) % 10 == 0:
                log_vram_usage(f"After {idx + 1} questions")

            # Print sample results (first 5)
            if idx < 5:
                print(f"\n{'='*80}")
                print(f"Question ID: {question_id}")
                print(f"Question: {question_text[:100]}...")
                print(f"Choices: {', '.join(choices)}")
                print(f"Model response: {response[:200]}...")
                print(f"Extracted answer: {answer}")
                print(f"{'='*80}\n")

        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            results.append({'id': question_id, 'answer': 'A', 'raw_response': f'Error: {str(e)}', 'prompt': prompt if 'prompt' in locals() else ''})

            # Clean up GPU memory even on error
            if 'pixel_values_gpu' in locals():
                del pixel_values_gpu
            torch.cuda.empty_cache()
            gc.collect()

    # Log final VRAM usage
    log_vram_usage("End of inference")
    print(f"\nTotal videos cached: {len(video_cache)}")

    return results

def save_results(results, output_dir, model_name):
    """Save results to CSV with timestamped filename"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract short model name
    model_short = model_name.split('/')[-1]  # e.g., InternVL3-8B

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename
    filename = f"submission_{model_short}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)

    # Write CSV with proper escaping
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'raw_response', 'prompt'])
        writer.writeheader()
        for result in results:
            writer.writerow({
                'id': result['id'],
                'answer': result['answer'],
                'raw_response': result.get('raw_response', ''),
                'prompt': result.get('prompt', '')
            })

    print(f"\nResults saved to: {output_path}")
    print(f"Total predictions: {len(results)}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Traffic Video QA Inference')
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='OpenGVLab/InternVL3-8B',
        choices=['OpenGVLab/InternVL3-8B', 'OpenGVLab/InternVL3-8B-Instruct'],
        help='Model to use for inference'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=8,
        help='Number of frames to extract from each video (default: 8)'
    )
    parser.add_argument(
        '--load_in_8bit',
        action='store_true',
        default=False,
        help='Enable 8-bit quantization for lower memory usage (default: False)'
    )
    parser.add_argument(
        '--use_grounding_dino',
        action='store_true',
        default=False,
        help='Enable Grounding DINO preprocessing to detect and crop traffic objects (default: False)'
    )
    parser.add_argument(
        '--grounding_dino_model',
        type=str,
        default='IDEA-Research/grounding-dino-tiny',
        help='Grounding DINO model to use (default: IDEA-Research/grounding-dino-tiny)'
    )
    parser.add_argument(
        '--detection_threshold',
        type=float,
        default=0.3,
        help='Detection confidence threshold for Grounding DINO (default: 0.3)'
    )
    parser.add_argument(
        '--max_detections_per_frame',
        type=int,
        default=2,
        help='Maximum number of detected objects to crop per frame (default: 2)'
    )
    parser.add_argument(
        '--max_num',
        type=int,
        default=3,
        help='Maximum number of patches per frame for dynamic preprocessing (default: 3, higher values = more detail but more VRAM)'
    )

    args = parser.parse_args()

    # Paths
    base_path = '../../RoadBuddy/traffic_buddy_train+public_test'
    json_path = os.path.join(base_path, 'public_test/public_test.json')
    output_dir = '../../demo2/internvl3_8B/output'

    print(f"\n{'='*80}")
    print("Traffic Video QA Inference")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Max patches per frame (max_num): {args.max_num}")
    print(f"Samples: {args.samples if args.samples else 'All'}")
    print(f"8-bit quantization: {'Enabled' if args.load_in_8bit else 'Disabled'}")
    print(f"Grounding DINO: {'Enabled' if args.use_grounding_dino else 'Disabled'}")
    if args.use_grounding_dino:
        print(f"  - Model: {args.grounding_dino_model}")
        print(f"  - Detection threshold: {args.detection_threshold}")
        print(f"  - Max detections/frame: {args.max_detections_per_frame}")
    print(f"{'='*80}\n")

    # Load test data
    questions = load_test_data(json_path, args.samples)

    # Load model
    model, tokenizer = load_model(args.model, args.load_in_8bit)
    log_vram_usage("After InternVL3 model loaded")

    # Load Grounding DINO if enabled
    grounding_dino_processor, grounding_dino_model = None, None
    if args.use_grounding_dino:
        grounding_dino_processor, grounding_dino_model = load_grounding_dino(
            model_id=args.grounding_dino_model
        )
        log_vram_usage("After Grounding DINO model loaded")

    # Run inference
    results = run_inference(
        model, tokenizer, questions, base_path, args.num_frames,
        grounding_dino_processor=grounding_dino_processor,
        grounding_dino_model=grounding_dino_model,
        detection_threshold=args.detection_threshold,
        max_detections_per_frame=args.max_detections_per_frame,
        max_num=args.max_num
    )

    # Save results
    output_path = save_results(results, output_dir, args.model)

    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
