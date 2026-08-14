"""
data_loader.py
---------------
Streams the VCR question/answer/rationale annotations and the matching
images from the Rowan/vcr dataset on Hugging Face, and builds an
{img_fn: PIL image} cache. Mirrors the notebook's "Load the VCR dataset" and
"Cache the needed images" sections.

Note from the notebook: the questions and images live separately in the
dataset, and multiple questions can point at the same image, e.g.
Image 1 -> Question 1, 2, 3
Image 2 -> Question 4, 5, 6, 7
Image 3 -> Question 8
...
So over N questions there can be far fewer unique images.
"""

from datasets import load_dataset

import config


def load_questions():
    """Stream the VCR question/answer/rationale annotations from Hugging
    Face. streaming=True fetches samples one at a time instead of
    downloading the whole (32GB) dataset up front."""
    return load_dataset(
        config.DATASET_NAME,
        name=config.QUESTIONS_CONFIG,
        split=config.SPLIT,
        streaming=config.STREAMING,
    )


def collect_samples(dataset, num_samples=config.NUM_SAMPLES):
    """Pull `num_samples` samples out of the stream into a plain Python list
    so they can be indexed/reused freely."""
    question_iter = iter(dataset)

    samples = []
    for _ in range(num_samples):
        samples.append(next(question_iter))

    print(len(samples))
    return samples


def load_images(split="train"):
    """Stream the image half of the dataset (different config tag,
    'image_examples')."""
    return load_dataset(
        config.DATASET_NAME,
        config.IMAGES_CONFIG,
        split=split,
        streaming=config.STREAMING,
    )


def build_image_cache(samples, image_dataset):
    """Build a lookup dict {img_fn: PIL image}, keeping only the images this
    batch needs, and stop early once they're all found."""
    needed_images = set(sample["img_fn"] for sample in samples)
    print(len(needed_images))

    image_cache = {}

    for img_sample in image_dataset:
        img_fn = img_sample["img_fn"]

        if img_fn in needed_images:
            image_cache[img_fn] = img_sample["image"]
            print("Loaded:", img_fn)

        if len(image_cache) == len(needed_images):
            break

    return image_cache
