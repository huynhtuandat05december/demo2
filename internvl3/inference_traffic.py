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
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm


def create_prompt(question, choices, video_prefix):
    SYSTEM_PROMPT = f"""
    Bạn là một AI chuyên gia phân tích an toàn giao thông. Nhiệm vụ duy nhất của bạn là phân tích video clip từ camera hành trình được cung cấp và trả lời một câu hỏi cụ thể về video đó.

    Nguyên tắc phân tích:

    Chỉ dựa vào hình ảnh: Câu trả lời của bạn chỉ được dựa trên những gì xuất hiện trực quan trong các khung hình của video.

    Tập trung vào đối tượng: Chú ý kỹ đến đèn giao thông, biển báo (giới hạn tốc độ, dừng, cảnh báo), vạch kẻ đường, các phương tiện khác (ô tô, xe tải, xe máy), người đi bộ, và điều kiện thời tiết/đường sá.

    Nhận thức về thời gian: Xem xét chuỗi sự kiện. Nếu câu hỏi về một hành động, hãy mô tả những gì xảy ra trong suốt clip.

    Tuân thủ định dạng: Đối với các câu hỏi trắc nghiệm, chỉ trả lời bằng chữ cái (ví dụ: A, B, C, D) của lựa chọn đúng. Không giải thích.

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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=8):
    """Load video and extract frames uniformly"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

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

def run_inference(model, tokenizer, questions, base_path, num_frames=8):
    """Run inference on all questions"""
    results = []
    video_cache = {}  # Cache video frames

    generation_config = dict(max_new_tokens=512, do_sample=False, temperature=0.0)

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
                pixel_values, num_patches_list = load_video(
                    full_video_path,
                    num_segments=num_frames,
                    max_num=1
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_cache[video_path] = (pixel_values, num_patches_list)
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                results.append({'id': question_id, 'answer': 'A', 'raw_response': f'Error: {str(e)}'})
                continue
        else:
            pixel_values, num_patches_list = video_cache[video_path]

        # Create video prefix with frame markers
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

        # Create full prompt
        prompt = create_prompt(question_text, choices, video_prefix)

        try:
            # Run inference
            response = model.chat(
                tokenizer,
                pixel_values,
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
                'raw_response': response
            })

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
            results.append({'id': question_id, 'answer': 'A', 'raw_response': f'Error: {str(e)}'})

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
        writer = csv.DictWriter(f, fieldnames=['id', 'answer', 'raw_response'])
        writer.writeheader()
        for result in results:
            writer.writerow({
                'id': result['id'],
                'answer': result['answer'],
                'raw_response': result.get('raw_response', '')
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
    print(f"Samples: {args.samples if args.samples else 'All'}")
    print(f"8-bit quantization: {'Enabled' if args.load_in_8bit else 'Disabled'}")
    print(f"{'='*80}\n")

    # Load test data
    questions = load_test_data(json_path, args.samples)

    # Load model
    model, tokenizer = load_model(args.model, args.load_in_8bit)

    # Run inference
    results = run_inference(model, tokenizer, questions, base_path, args.num_frames)

    # Save results
    output_path = save_results(results, output_dir, args.model)

    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
