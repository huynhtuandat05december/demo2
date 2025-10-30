# Demo2 - Video Understanding Models Inference

Video analysis and Q&A using OpenGVLab's vision-language models for video understanding tasks on the RoadBuddy traffic dataset.

Supported models:
- **InternVideo2.5 Chat 8B** - Video-specific multimodal model
- **InternVL3-8B** - General vision-language model with video support
- **InternVL3-8B-Instruct** - Instruction-tuned variant

## Overview

This project provides inference capabilities for video analysis using multiple vision-language models. It supports:
- **Traffic video question answering** for dashcam footage
- **Batch processing** of multiple videos with progress tracking
- **8-bit quantization** to fit in 24GB VRAM
- **Automatic answer parsing** (A/B/C/D extraction)
- **CSV output** in competition submission format
- Dynamic frame sampling based on video duration
- Adaptive image preprocessing for optimal performance

### Model Comparison

| Model | VRAM (8-bit) | VRAM (Full) | Speed | Best For |
|-------|--------------|-------------|-------|----------|
| InternVL3-8B | ~10-12GB | ~16-20GB | Fast | General vision tasks, less VRAM |
| InternVL3-8B-Instruct | ~10-12GB | ~16-20GB | Fast | Instruction following |
| InternVideo2.5 Chat 8B | ~24GB | ~40GB+ | Medium | Video-specific tasks |

**Quick recommendation:** Use InternVL3-8B for lower VRAM usage and faster inference.

## Requirements

- Python >= 3.10
- CUDA-compatible GPU (24GB VRAM recommended with 8-bit quantization)
- Dependencies listed in `pyproject.toml`

## Installation

### 1. Install the project dependencies

```bash
cd demo2
pip install -e .
```

### 2. Install flash-attn separately (if needed)

Due to build requirements, `flash-attn` may need special installation:

```bash
pip install flash-attn --no-build-isolation
```

### Dependencies

- `transformers==4.40.1` - Hugging Face transformers library
- `av` - Video processing
- `imageio` - Image I/O operations
- `decord` - Efficient video loading
- `opencv-python` - Computer vision operations
- `flash-attn` - Optimized attention mechanism
- `bitsandbytes` - 8-bit quantization support
- `tqdm` - Progress bars
- `pandas` - CSV handling

## Model

The project uses the **InternVideo2.5 Chat 8B** model from OpenGVLab:
- Model ID: `OpenGVLab/InternVideo2_5_Chat_8B`
- Automatically downloaded from Hugging Face Hub on first run (~16GB download)
- Supports optional 8-bit quantization for reduced VRAM usage

## Usage

### Quick Start: RoadBuddy Dataset Inference

Run inference on the RoadBuddy public test dataset with 8-bit quantization:

```bash
cd internvideo_2_5_8B
python run_inference.py --load_in_8bit
```

This will:
1. Load the InternVideo2.5 model with 8-bit quantization
2. Process all 405 test videos from the RoadBuddy dataset
3. Generate a timestamped CSV file with predictions

### Command-Line Options

```bash
python run_inference.py [OPTIONS]

Options:
  --load_in_8bit           Enable 8-bit quantization (recommended for 24GB VRAM)
  --input_json PATH        Path to test JSON file (default: RoadBuddy public_test.json)
  --data_dir PATH          Base directory containing videos (default: RoadBuddy data directory)
  --output_dir PATH        Output directory for CSV (default: ../output/)
  --num_segments INT       Number of frames to sample (default: 128)
  --samples INT            Number of samples to process (default: all samples)
  --model_path PATH        HuggingFace model ID or local path
```

### Examples

**With 8-bit quantization (24GB VRAM):**
```bash
python run_inference.py --load_in_8bit
```

**Without quantization (40GB+ VRAM required):**
```bash
python run_inference.py
```

**Custom paths:**
```bash
python run_inference.py \
  --load_in_8bit \
  --input_json /path/to/test.json \
  --data_dir /path/to/videos \
  --output_dir /path/to/output
```

**Adjust frame sampling for speed/quality tradeoff:**
```bash
# Faster inference, less accurate
python run_inference.py --load_in_8bit --num_segments 64

# Slower inference, more accurate
python run_inference.py --load_in_8bit --num_segments 256
```

**Test with limited samples (useful for debugging):**
```bash
# Test with just 1 sample
python run_inference.py --load_in_8bit --samples 1

# Test with 10 samples
python run_inference.py --load_in_8bit --samples 10

# Process all samples (default)
python run_inference.py --load_in_8bit
```

### Output Format

The script generates a CSV file with the following format:

**Filename:** `output/public_test_InternVideo2_5_8B_YYYYMMDD_HHMMSS.csv`

**Content:**
```csv
id,answer
testa_0001,A
testa_0002,B
testa_0003,D
...
```

### Python API Usage

You can also use the inference class directly in Python:

```python
from inference_internvideo_2_5_8B import InternVideo2Inference
from datetime import datetime

# Initialize model with 8-bit quantization
inferencer = InternVideo2Inference(
    model_path='OpenGVLab/InternVideo2_5_Chat_8B',
    load_in_8bit=True
)

# Run complete pipeline (all samples)
inferencer.run_pipeline(
    test_json_path='path/to/public_test.json',
    data_dir='path/to/videos/',
    output_csv_path='output/results.csv',
    num_segments=128
)

# Or test with limited samples
inferencer.run_pipeline(
    test_json_path='path/to/public_test.json',
    data_dir='path/to/videos/',
    output_csv_path='output/results.csv',
    num_segments=128,
    max_samples=10  # Only process first 10 samples
)

# Or run single inference
answer = inferencer.infer(
    video_path='path/to/video.mp4',
    question='Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?',
    choices=['A. Không có thông tin', 'B. Dầu Giây Long Thành', 'C. Đường Đỗ Xuân Hợp', 'D. Xa Lộ Hà Nội'],
    num_segments=128
)
print(f"Answer: {answer}")
```

## InternVL3 Models

### Quick Start: InternVL3-8B Inference

Run inference on the RoadBuddy public test dataset using InternVL3-8B:

```bash
cd internvl3_8B
python inference_traffic.py
```

### Command-Line Options

```bash
python inference_traffic.py [OPTIONS]

Options:
  --samples INT            Number of samples to process (default: all 405)
  --model MODEL_NAME       Model to use for inference
                          Choices: OpenGVLab/InternVL3-8B (default)
                                   OpenGVLab/InternVL3-8B-Instruct
  --num_frames INT         Number of frames to extract per video (default: 8)
  --load_in_8bit          Enable 8-bit quantization (default: disabled)
```

### Examples

**Test with 5 samples (quick test):**
```bash
python inference_traffic.py --samples 5
```

**Use instruction-tuned model:**
```bash
python inference_traffic.py --samples 5 --model OpenGVLab/InternVL3-8B-Instruct
```

**Full run with 8-bit quantization:**
```bash
python inference_traffic.py --load_in_8bit
```

**Full run with instruct model and 8-bit quantization:**
```bash
python inference_traffic.py --load_in_8bit --model OpenGVLab/InternVL3-8B-Instruct
```

**Custom frame count (more frames = better quality, slower):**
```bash
python inference_traffic.py --samples 10 --num_frames 16
```

**Debug single sample:**
```bash
python inference_traffic.py --samples 1 --load_in_8bit
```

### Output Format

**Location:** `demo2/internvl3_8B/output/`

**Filename:** `submission_InternVL3-8B_YYYYMMDD_HHMMSS.csv`

Example: `submission_InternVL3-8B_20251030_143052.csv`

**Content:**
```csv
id,answer
testa_0001,B
testa_0002,A
testa_0003,D
...
```

### InternVL3 Features

**Frame Extraction:**
- Extracts 8 uniform frames by default (configurable)
- Uses dynamic preprocessing for optimal aspect ratio
- Caches frames for videos with multiple questions
- Thumbnail mode enabled for better context

**Model Support:**
- Base model: `OpenGVLab/InternVL3-8B`
- Instruct model: `OpenGVLab/InternVL3-8B-Instruct`
- Optional 8-bit quantization for reduced VRAM
- Multi-GPU support with automatic layer splitting

**Vietnamese Q&A:**
- Custom prompt template for traffic questions
- Handles 2-choice (Yes/No) and 4-choice questions
- Robust answer extraction with regex patterns
- Fallback to 'A' if parsing fails

**Memory Requirements:**
| Configuration | VRAM Required |
|--------------|---------------|
| Full precision | ~16-20GB |
| 8-bit quantization | ~10-12GB |

## Configuration

### Memory Requirements

| Configuration | VRAM Required | Speed | Accuracy |
|--------------|---------------|-------|----------|
| 8-bit quantization | ~24GB | Normal | High |
| Full precision (bfloat16) | ~40GB+ | Normal | High |

**Recommendation:** Use `--load_in_8bit` flag for most setups with 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000).

### Generation Settings

Default generation configuration (`inference_internvideo_2_5_8B.py:300-306`):

```python
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)
```

### Video Processing

- **Frame sampling**: Configurable via `--num_segments` (default: 128)
- **Input size**: 448x448 pixels
- **Dynamic preprocessing**: Automatically adjusts to video aspect ratio
- **Thumbnail mode**: Enabled for better context
- **CUDA cache clearing**: Automatic between videos to prevent OOM

## Project Structure

```
demo2/
├── README.md
├── pyproject.toml
├── main.py
├── output/                              # Generated CSV files
├── internvideo_2_5_8B/
│   ├── inference_internvideo_2_5_8B.py  # Core inference class
│   └── run_inference.py                 # CLI entry point
└── internvl3_8B/
    └── inference_traffic.py             # InternVL3-8B inference script
```

## Features

### Vietnamese Prompt Template

The model uses an optimized Vietnamese prompt template for dashcam video analysis:

```
Đây là video từ camera hành trình của xe ô tô đang di chuyển trên đường.
Hãy phân tích video và trả lời câu hỏi sau:

Câu hỏi: {question}

Các lựa chọn:
{choices}

Hãy chọn đáp án đúng nhất và chỉ trả lời bằng chữ cái A, B, C, hoặc D.
```

### Answer Parsing

The system uses multiple regex patterns to extract answers:
- Single letter patterns: `A`, `B`, `C`, `D`
- Vietnamese answer patterns: "Đáp án: A", "Chọn: B", "Trả lời: C"
- Falls back to 'A' if parsing fails (with warning)

### Error Handling

- **Missing videos**: Defaults to 'A' with warning
- **Inference errors**: Catches exceptions, logs error, continues processing
- **Progress tracking**: Real-time progress bar with tqdm
- **CUDA memory management**: Automatic cache clearing between videos

## Troubleshooting

### CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Enable 8-bit quantization: `--load_in_8bit`
2. Reduce frame sampling: `--num_segments 64`
3. Close other GPU processes
4. Use a GPU with more VRAM

### Slow Inference

**Solutions:**
1. Reduce `--num_segments` (e.g., 64 or 96)
2. Use 8-bit quantization (slightly faster)
3. Ensure videos are on local SSD (not network drive)

### Installation Issues

**flash-attn compilation errors:**
```bash
pip install flash-attn --no-build-isolation
```

**CUDA toolkit not found:**
- Install CUDA 11.8 or 12.1
- Ensure `nvcc --version` works

**Python version mismatch:**
- Ensure Python 3.10 or higher: `python --version`

### Answer Parsing Warnings

If you see: `Warning: Could not parse answer from output`

This means the model's response didn't match expected patterns. The system defaults to 'A'. You can:
1. Check the model output in console
2. Adjust the prompt template in `build_prompt()` method
3. Add more parsing patterns in `parse_answer()` method

## Performance

**Expected Performance on RoadBuddy Public Test (405 videos):**
- **Time**: ~2-4 hours (depends on GPU, num_segments)
- **Throughput**: ~2-3 videos/minute with `num_segments=128`
- **VRAM Usage**: ~22GB with 8-bit quantization

**Tips for Faster Inference:**
- Use `--num_segments 64` for 2x speedup (may reduce accuracy)
- Ensure videos are on local SSD
- Use latest CUDA version for better performance

## Notes

- The model automatically downloads on first run (~16GB)
- Video processing uses decord for efficient loading
- Supports Vietnamese and English questions
- Progress and errors are logged to console
- Each inference clears CUDA cache to prevent memory leaks

## License

Refer to the InternVideo2.5 model license and terms of use from OpenGVLab.
