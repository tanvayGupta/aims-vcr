"""
config.py
---------
Central place for the constants that were scattered across the notebook's
cells: which model to load, how to quantize it, which dataset/config to
stream from, and how many samples to evaluate on.
"""

import torch

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Qwen2.5-VL-3B-Instruct, loaded in 4-bit precision to fit a 16GB T4 GPU.
# (The notebook also had a commented-out Qwen3-VL-4B-Instruct block — see
# model_loader.py for the equivalent, kept as an easy opt-in swap.)
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

QUANTIZATION_KWARGS = dict(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

DEVICE_MAP = "auto"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "Rowan/vcr"
QUESTIONS_CONFIG = "questions"
IMAGES_CONFIG = "image_examples"
SPLIT = "train"
STREAMING = True

# Number of question samples to pull from the stream and evaluate on.
NUM_SAMPLES = 50

# thumbnail() target size (keeps aspect ratio, longest side capped at this).
IMAGE_THUMBNAIL_SIZE = (768, 768)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 1  # the whole reply is a single letter (A-D)
DO_SAMPLE = False   # greedy decoding == temperature -> 0
