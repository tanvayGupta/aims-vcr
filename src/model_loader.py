"""
model_loader.py
----------------
Loads Qwen2.5-VL-3B-Instruct in 4-bit precision (fits on a single T4 GPU)
plus its processor. Mirrors the notebook's "Load the model" cell.
"""

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

import config


def check_gpu():
    """Sanity check: confirm PyTorch can see a GPU before loading a
    multi-billion-parameter model."""
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


def load_model():
    """Load Qwen2.5-VL-3B-Instruct in 4-bit precision plus its processor.

    Returns:
        (model, processor)
    """
    quantization_config = BitsAndBytesConfig(**config.QUANTIZATION_KWARGS)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        quantization_config=quantization_config,
        device_map=config.DEVICE_MAP,
    )

    processor = AutoProcessor.from_pretrained(config.MODEL_NAME)

    print("Model loaded!")
    return model, processor


# ---------------------------------------------------------------------------
# HIDDEN CODE: ACTIVATES QWEN3VL 4B model shhhhhhhhh, BUT YOU HAVE TO
# UNCOMMENT IT OBV
# This takes roughly 8-9 Gigs (RAM+VRAM). Answer generation also roughly
# takes 8 times the time, felt like 3 business days.
# ---------------------------------------------------------------------------
# from transformers import Qwen3VLForConditionalGeneration
#
# def load_model_qwen3vl():
#     model_name = "Qwen/Qwen3-VL-4B-Instruct"
#
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#     )
#
#     model = Qwen3VLForConditionalGeneration.from_pretrained(
#         model_name,
#         quantization_config=quantization_config,
#         device_map="auto",
#     )
#
#     processor = AutoProcessor.from_pretrained(model_name)
#
#     print("Model loaded!")
#     return model, processor
