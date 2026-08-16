# Qwen3-VL-4B Fine-Tuning on Visual Commonsense Reasoning (VCR)

# Please switch from the branch main to aims

LoRA fine-tuning pipeline adapting `Qwen/Qwen3-VL-4B-Instruct` for the VCR task (answer-selection + rationale-selection) on the `Rowan/vcr` dataset. Built as part of the AIMS recruitment task, Round 2.

- **HF Model (LoRA adapter):** [zzephyrr/qwen3vl-vcr-lora](https://huggingface.co/zzephyrr/qwen3vl-vcr-lora)
- **W&B Project:** [tanygupt360-delhi-technological-university/huggingface](https://wandb.ai/tanygupt360-delhi-technological-university/huggingface)
- **Hardware:** Single NVIDIA T4 (Google Colab)

## Overview

The pipeline fine-tunes only the language decoder of Qwen3-VL-4B via LoRA, leaving the vision encoder and vision-language merger frozen, and trains on both stages of the VCR task — selecting the correct answer, and separately selecting the correct rationale for that answer.

## Dataset

`Rowan/vcr` splits questions and images across two separate configs, joined via `img_fn`:

| Config | Contents |
|---|---|
| `questions` | question text, answer/rationale choices, correct labels |
| `image_examples` | the image, object bounding boxes, object tags |

Images are joined to their questions via `img_fn`, resized to a 768×768 thumbnail bound, and annotated with bounding boxes + object index labels to support grounded reasoning in the prompt.

## Model & LoRA Configuration

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Precision | bf16 (matches model's native dtype) |
| LoRA rank / alpha | r=8, α=16 |
| LoRA dropout | 0.067 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Frozen | Vision encoder (SigLIP2 ViT), vision-language merger, embeddings, LM head |
| Trainer | TRL `SFTTrainer` |
| Optimizer | AdamW (`adamw_torch`) |
| Learning rate | 2e-4 |

## Training Strategy

Training was run in three staged passes on the same LoRA adapter, using non-overlapping slices of `Rowan/vcr` at each stage to avoid re-training on already-seen examples:

1. **Stage 1** — Answer-selection only, 200 questions.
2. **Stage 2** — Rationale-selection introduced as a second objective (built from ground-truth answers), closing a gap where the adapter had only seen the answer-selection task.
3. **Stage 3** — Fresh 75-question slice (`train[200:275]`), interleaving answer + rationale examples, continued on the same adapter.

Loss trend across stages: **13.53 → 10.40 → 7.71 → ~3.8**, tracked live in W&B.

## Key Bugs Found & Fixed

| Issue | Fix |
|---|---|
| `GradScaler` incompatible with bf16 weights | Switched fully to bf16 (no loss scaling needed), dropped fp16 |
| Images nested inside `messages.content` decoded as raw dict, not PIL Image | Moved images to a top-level `"images"` column, matched positionally to `{"type": "image"}` placeholders |
| Model trained to output full answer **text** instead of the letter the eval prompt asks for | Fixed training target to the answer letter (`A`–`D`), matching the inference prompt format |
| Rationale-selection never trained | Added a matched rationale training objective using ground-truth answers |

## Evaluation

Evaluated on a held-out slice of `Rowan/vcr` disjoint from all training data, scoring answer-selection and rationale-selection accuracy independently.

Note: This shows the results of the best model only, not the baselines, check out the v1_documented.ipynb for more detailson all models

| Metric | Result |
|---|---|
| Answer-selection accuracy | 66% |
| Rationale-selection accuracy | 48% |
| Answer and Rationale accuracy | 30% |

## Experiment Tracking & Publishing

- **Weights & Biases**: every training stage logged under a distinct run name (`report_to="wandb"`); loss/system metrics stream live, so full run history survived independent of the Colab session's lifetime.
- **HuggingFace Hub**: adapter published to `zzephyrr/qwen3vl-vcr-lora` with `push_to_hub=True` + `save_strategy="steps"`, so checkpoints were committed throughout training, not just at the end. Load with:

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

base = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
model = PeftModel.from_pretrained(base, "zzephyrr/qwen3vl-vcr-lora")
```

## Repo Contents

- `v9FINAL.ipynb` — full training + evaluation notebook
- `VCR_FineTuning_Report.docx` — written report (architecture, strategy, results, design choices)
- `predictions.json` — model predictions on the test set
