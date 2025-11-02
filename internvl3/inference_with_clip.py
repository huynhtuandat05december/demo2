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
from transformers import AutoModel, AutoTokenizer, AutoConfig, CLIPProcessor, CLIPModel
from tqdm import tqdm
import gc
import cv2


def log_vram_usage(stage=""):
    """Log current VRAM usage for monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {stage}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
    return allocated if torch.cuda.is_available() else 0


def load_clip_model(model_id="openai/clip-vit-base-patch32", device=None):
    """Load CLIP model for frame selection"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading CLIP model: {model_id}")
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    clip_model = CLIPModel.from_pretrained(model_id).to(device)
    clip_model.eval()
    print("CLIP model loaded successfully!")

    return clip_processor, clip_model


@torch.no_grad()
def get_key_frames_clip(video_path, question_text, clip_processor, clip_model, n_frames=16):
    """
    Extract key frames from video using CLIP similarity scoring

    Args:
        video_path: Path to video file
        question_text: Question text to match frames against
        clip_processor: CLIP processor
        clip_model: CLIP model
        n_frames: Number of top frames to select

    Returns:
        list of PIL Images: Top N most relevant frames
    """
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return []

    # Extract all frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    all_frames_pil = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames_pil.append(Image.fromarray(frame_rgb))
    cap.release()

    if not all_frames_pil:
        print(f"No frames extracted from video: {video_path}")
        return []

    # Limit frames if too many to avoid OOM (max ~15s at 30fps)
    if len(all_frames_pil) > 450:
        indices = np.linspace(0, len(all_frames_pil) - 1, 450, dtype=int)
        all_frames_pil = [all_frames_pil[i] for i in indices]

    print(f"Extracted {len(all_frames_pil)} frames. Running CLIP scoring...")

    # Process frames in batches to save VRAM
    batch_size = 64
    all_scores = []
    device = clip_model.device

    # Encode text once (shared across all frames)
    text_inputs = clip_processor(
        text=[question_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    text_features = clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # Score frames in batches
    for i in range(0, len(all_frames_pil), batch_size):
        batch_images = all_frames_pil[i:i + batch_size]
        image_inputs = clip_processor(
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        image_features = clip_model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity (scaled by 100)
        similarity = (100.0 * image_features @ text_features.T).squeeze(0)
        all_scores.extend(similarity.cpu().numpy())

    # Select top N frames by score
    top_n_indices = np.argsort(all_scores)[-n_frames:]
    top_n_indices = sorted(top_n_indices)  # Sort chronologically

    key_frames = [all_frames_pil[int(i)] for i in top_n_indices]
    print(f"Selected {len(key_frames)} key frames with CLIP.")

    return key_frames


def create_prompt(question, choices, video_prefix):
    """Create prompt for Vietnamese traffic video Q&A"""
    SYSTEM_PROMPT = f"""
    B¡n là mÙt AI chuyên gia phân tích an toàn giao thông. NhiÇm vå duy nh¥t cça b¡n là phân tích video clip të camera hành trình °ãc cung c¥p và tr£ lÝi mÙt câu hÏi cå thÃ vÁ video ó.

    Nguyên t¯c phân tích:

    ChÉ dña vào hình £nh: Câu tr£ lÝi cça b¡n chÉ °ãc dña trên nhïng gì xu¥t hiÇn trñc quan trong các khung hình cça video.

    T­p trung vào Ñi t°ãng: Chú ý kù ¿n èn giao thông, biÃn báo (giÛi h¡n tÑc Ù, dëng, c£nh báo), v¡ch k» °Ýng, các ph°¡ng tiÇn khác (ô tô, xe t£i, xe máy), ng°Ýi i bÙ, và iÁu kiÇn thÝi ti¿t/°Ýng sá.

    Nh­n théc vÁ thÝi gian: Xem xét chu×i sñ kiÇn. N¿u câu hÏi vÁ mÙt hành Ùng, hãy mô t£ nhïng gì x£y ra trong suÑt clip.

    Tuân thç Ënh d¡ng: Ñi vÛi các câu hÏi tr¯c nghiÇm, chÉ tr£ lÝi b±ng chï cái (ví då: A, B, C, D) cça lña chÍn úng. Không gi£i thích.

    {video_prefix}

    Câu hÏi: {question}

    Các lña chÍn: {choices}

    ChÉ tr£ lÝi b±ng chï cái (A, B, C, ho·c D) t°¡ng éng vÛi lña chÍn úng.
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


def load_video_from_frames(frames, input_size=448, max_num=3):
    """
    Load video from pre-selected frames (CLIP-selected)

    Args:
        frames: List of PIL Images
        input_size: Size for image preprocessing
        max_num: Max number of patches per frame

    Returns:
        pixel_values: Tensor of processed frames
        num_patches_list: List of patch counts per frame
    """
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    for img in frames:
        # Process frame with dynamic preprocessing
        img_patches = dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)

        # Transform all patches
        pixel_values = [transform(tile) for tile in img_patches]
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

    # Enable gradient checkpointing to reduce activation memory by 40-50%
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled for memory optimization")

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


def run_inference(model, tokenizer, questions, base_path, clip_processor, clip_model,
                  num_frames=16, max_num=3):
    """Run inference on all questions using CLIP frame selection"""
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
        cache_key = f"{video_path}_{question_text}"  # Cache per video+question for CLIP

        if cache_key not in video_cache:
            try:
                # Extract key frames using CLIP
                key_frames = get_key_frames_clip(
                    full_video_path,
                    question_text,
                    clip_processor,
                    clip_model,
                    n_frames=num_frames
                )

                if not key_frames:
                    print(f"No frames extracted for {video_path}")
                    results.append({'id': question_id, 'answer': 'A', 'raw_response': 'Error: No frames extracted'})
                    continue

                # Process frames for InternVL3
                pixel_values, num_patches_list = load_video_from_frames(
                    key_frames,
                    input_size=448,
                    max_num=max_num
                )

                # Store on CPU to save VRAM - will move to GPU only during inference
                pixel_values = pixel_values.to(torch.bfloat16).cpu()
                video_cache[cache_key] = (pixel_values, num_patches_list)

            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                results.append({'id': question_id, 'answer': 'A', 'raw_response': f'Error: {str(e)}'})
                continue
        else:
            pixel_values, num_patches_list = video_cache[cache_key]

        # Create video prefix with frame markers
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

        # Create full prompt
        prompt = create_prompt(question_text, choices, video_prefix)

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
    print(f"\nTotal video+question pairs cached: {len(video_cache)}")

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
    filename = f"submission_{model_short}_clip_{timestamp}.csv"
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
    parser = argparse.ArgumentParser(description='Traffic Video QA Inference with CLIP Frame Selection')
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
        default=16,
        help='Number of top frames to select from each video using CLIP (default: 16)'
    )
    parser.add_argument(
        '--load_in_8bit',
        action='store_true',
        default=False,
        help='Enable 8-bit quantization for lower memory usage (default: False)'
    )
    parser.add_argument(
        '--clip_model',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='CLIP model to use for frame selection (default: openai/clip-vit-base-patch32)'
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
    print("Traffic Video QA Inference with CLIP Frame Selection")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Top frames to select: {args.num_frames}")
    print(f"Max patches per frame (max_num): {args.max_num}")
    print(f"Samples: {args.samples if args.samples else 'All'}")
    print(f"8-bit quantization: {'Enabled' if args.load_in_8bit else 'Disabled'}")
    print(f"{'='*80}\n")

    # Load test data
    questions = load_test_data(json_path, args.samples)

    # Load CLIP model
    clip_processor, clip_model = load_clip_model(model_id=args.clip_model)
    log_vram_usage("After CLIP model loaded")

    # Load InternVL3 model
    model, tokenizer = load_model(args.model, args.load_in_8bit)
    log_vram_usage("After InternVL3 model loaded")

    # Run inference
    results = run_inference(
        model, tokenizer, questions, base_path,
        clip_processor, clip_model,
        num_frames=args.num_frames,
        max_num=args.max_num
    )

    # Save results
    output_path = save_results(results, output_dir, args.model)

    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
