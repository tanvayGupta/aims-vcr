# Aims-VCR

Zero-shot Visual Commonsense Reasoning with Qwen2.5-VL-3B-Instruct: a
two-stage pipeline that (1) picks the best-supported answer to a question
about an image, then (2) picks the rationale that best explains that
answer, over `config.NUM_SAMPLES` streamed VCR samples.

This is a straight conversion of `version1_documented.ipynb` into a runnable
project — no logic added or removed, just split across linked modules so
each part (model, data, prompts, inference, orchestration) lives in its own
file.

## Files

| File               | Notebook section(s) it came from                                   |
|---------------------|---------------------------------------------------------------------|
| `config.py`          | Constants used throughout (model name, quantization, dataset, sample count) |
| `state.py`           | "Shared state" cell (module-level globals: `output`, `correct_answers`, etc.) |
| `model_loader.py`    | "Setup" + "Load the model" (GPU check, Qwen2.5-VL 4-bit load; Qwen3-VL alt kept commented) |
| `data_loader.py`     | "Load the VCR dataset" + "Cache the needed images" |
| `prompts.py`         | "Prompt builders" — the improved `aPrompt`/`rPrompt` (the versions in effect after the notebook's later "improved prompt" cells) |
| `inference.py`       | "Run inference" — `output_generator()` |
| `main.py`            | "Try the pipeline on one sample" + "Evaluate all samples" — orchestrates everything in the same order the notebook cells ran |

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with enough VRAM for 4-bit Qwen2.5-VL-3B (the notebook
used a T4 with 16GB; expect the model to load in ~6-7GB and spike to
~12-13GB during generation).

## Run

```bash
python main.py
```

This will:
1. Check for a GPU and load Qwen2.5-VL-3B-Instruct in 4-bit.
2. Stream 50 question samples and their images from `Rowan/vcr`.
3. Run a single sanity-check sample through both stages.
4. Run the full two-stage (answer, then rationale) evaluation loop over all
   samples.
5. Print the raw Q->A and QA->R correct counts.

## Results (from the notebook)

### Qwen2.5-VL-3B-Instruct, 4-bit, original prompt
| Metric  | Accuracy |
|---------|----------|
| Q->A    | 26%      |
| QA->R   | 32%      |
| Q->AR   | 6%       |

### Qwen3-VL-4B-Instruct, 4-bit, original prompt
| Metric  | Accuracy | Improvement |
|---------|----------|-------------|
| Q->A    | 32%      | 23.1%       |
| QA->R   | 36%      | 12.5%       |
| Q->AR   | 10%      | 66.7%       |

### Qwen3-VL-4B-Instruct, 4-bit, improved prompt (the version shipped in `prompts.py`)
| Metric  | Accuracy | Improvement (vs Qwen2.5 base) |
|---------|----------|-------------------------------|
| Q->A    | 23.2%    | -2.8%                         |
| QA->R   | 23.2%    | -8.8%                         |
| Q->AR   | 7%       | 66.7%                         |

To switch to Qwen3-VL-4B, uncomment `load_model_qwen3vl()` in
`model_loader.py` and call it instead of `load_model()` in `main.py`.
