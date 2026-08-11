# run.py

from src.model import load_model
from src.vcr_dataset import load_questions
from src.inference import build_answer_prompt
from src.evaluate import answer_correct


def main():

    # Load model
    model, processor = load_model()

    # Load dataset
    dataset = load_questions()

    # Get one sample
    sample = next(iter(dataset))

    # Build prompt
    prompt = build_answer_prompt(sample)

    print(prompt)

    # We'll add actual Qwen inference here next.


if __name__ == "__main__":
    main()