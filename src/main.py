"""
main.py
-------
Entry point that wires every module together and reproduces the notebook's
end-to-end run, in the same order the cells executed:

1. GPU sanity check + load the model/processor.
2. Stream `config.NUM_SAMPLES` question samples from the VCR dataset.
3. Cache the unique images those samples reference.
4. Sanity-check the pipeline on a single sample.
5. Run Stage 1 (answer selection) + Stage 2 (rationale selection) over every
   sample, tallying accuracy as it goes.
6. Print the final Q->A and QA->R raw counts.
"""

import config
import state
from model_loader import check_gpu, load_model
from data_loader import load_questions, collect_samples, load_images, build_image_cache
from prompts import aPrompt, rPrompt
from inference import output_generator


def run_single_sample_check(model, processor, samples, image_cache, index=8):
    """Pick one sample to build and sanity-check the two-stage pipeline on,
    before the full run over every sample."""
    sample = samples[index]

    image = image_cache[sample["img_fn"]]
    print(image.size)

    # thumbnail() resizes in place (keeping aspect ratio) so the longest
    # side is at most 768px
    image.thumbnail(config.IMAGE_THUMBNAIL_SIZE)
    print(image.size)

    state.image = image

    prompt_a = aPrompt(sample)
    output_generator(model, processor, prompt_a, True, sample)

    prompt_r = rPrompt(sample)
    output_generator(model, processor, prompt_r, False, sample)


def evaluate_all_samples(model, processor, samples, image_cache):
    """Run both stages over all samples, tallying
    state.correct_answers / state.correct_rationales as we go."""
    for sample in samples:
        image = image_cache[sample["img_fn"]]
        image.thumbnail(config.IMAGE_THUMBNAIL_SIZE)
        state.image = image

        # Stage 1: build the answer prompt and generate a prediction
        prompt_a = aPrompt(sample)
        output_generator(model, processor, prompt_a, True, sample)

        # Stage 2: build the rationale prompt (reads Stage 1's predicted
        # letter) and generate a prediction
        prompt_r = rPrompt(sample)
        output_generator(model, processor, prompt_r, False, sample)

    # raw counts out of len(samples) — e.g. correct_answers / len(samples)
    # gives an accuracy percentage
    print(state.correct_answers)
    print(state.correct_rationales)


def main():
    # --- Setup ---
    check_gpu()
    model, processor = load_model()

    # --- Load the VCR dataset ---
    dataset = load_questions()
    samples = collect_samples(dataset, config.NUM_SAMPLES)

    # --- Cache the needed images ---
    image_dataset = load_images("train")
    image_cache = build_image_cache(samples, image_dataset)

    # --- Try the pipeline on one sample ---
    run_single_sample_check(model, processor, samples, image_cache, index=8)

    # --- Evaluate all samples ---
    evaluate_all_samples(model, processor, samples, image_cache)

    total = len(samples)
    print(f"Q->A accuracy: {state.correct_answers}/{total}")
    print(f"QA->R accuracy: {state.correct_rationales}/{total}")


if __name__ == "__main__":
    main()
