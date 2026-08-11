from datasets import load_dataset


def load_questions(split="train"):
    return load_dataset(
        "Rowan/vcr",
        "questions",
        split=split,
        streaming=True,
    )


def load_images(split="train"):
    return load_dataset(
        "Rowan/vcr",
        "image_examples",
        split=split,
        streaming=True,
    )


def get_samples(dataset, n):
    iterator = iter(dataset)

    return [
        next(iterator)
        for _ in range(n)
    ]