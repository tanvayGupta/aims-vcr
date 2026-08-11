def build_answer_prompt(sample):
    question = sample["question_text"]
    answers = sample["answer_choice_texts"]

    return f"""
You are solving a Visual Commonsense Reasoning (VCR)
multiple-choice question.

Question:
{question}

Answer choices:

A. {answers[0]}
B. {answers[1]}
C. {answers[2]}
D. {answers[3]}

Select the answer that is best supported by the image
and question.

Respond with ONLY one letter: A, B, C, or D.
""".strip()

def build_rationale_prompt(sample, predicted_answer):
    question = sample["question_text"]
    rationales = sample["rationale_choice_texts"]

    return f"""
You are solving the rationale-selection stage of a
Visual Commonsense Reasoning (VCR) task.

Question:
{question}

Selected answer:
{predicted_answer}

Rationale choices:

A. {rationales[0]}
B. {rationales[1]}
C. {rationales[2]}
D. {rationales[3]}

Choose the rationale that best explains the selected answer.

Respond with ONLY one letter: A, B, C, or D.
""".strip()