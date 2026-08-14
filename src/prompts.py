"""
prompts.py
----------
Two functions that format a VCR sample into the text prompt for each stage.

The notebook defined aPrompt/rPrompt twice: a first pass (cells 22-23), then
an improved version later (cells 30-31) after the "What next then?" section.
Since a top-to-bottom run of the notebook ends with the improved versions in
effect, those are the ones reproduced here.
"""

import state


def aPrompt(sample):
    """Build the Stage 1 prompt: ask the model to pick the answer (A-D)
    best supported by this VCR sample's image and question text."""

    question = sample["question_text"]
    answers = sample["answer_choice_texts"]

    prompt = f"""
You are solving a Visual Commonsense Reasoning (VCR) multiple-choice question.

Your task is to identify the ONE answer choice that is best supported by the image and the question.

Follow this reasoning process internally:

1. Carefully inspect the image.
2. Identify the people, objects, actions, relationships, and relevant visual details.
3. Determine exactly what the question is asking.
4. Evaluate each answer choice against the visual evidence and the question.
5. Reject choices that contradict the image, answer a different question, or rely on unsupported assumptions.
6. Select the choice with the strongest evidence.

Important:
- Base your decision primarily on evidence visible in the image.
- Use ordinary commonsense only when it is necessary to interpret the situation.
- Do not choose an answer merely because it sounds plausible.
- Pay close attention to the distinction between similar people, objects, actions, and relationships.
- There is exactly ONE correct answer.

Question:
{question}

Answer choices:

A. {answers[0]}
B. {answers[1]}
C. {answers[2]}
D. {answers[3]}

After reasoning internally, respond with ONLY the letter of the correct choice:
A, B, C, or D.
"""
    return prompt
    # print(prompt)


def rPrompt(sample):
    """Build the Stage 2 prompt: ask the model to pick the rationale (A-D)
    that best explains the answer predicted by the most recent
    output_generator() call. Reads the Stage 1 result from the module-level
    `state.output` list rather than taking it as a function argument."""

    # `state.output` was last set by output_generator() when Stage 1 ran;
    # output[0] is the model's raw one-letter reply (e.g. "A") — so
    # predicted_answer is that letter, not the text of the answer choice.
    state.predicted_answer = state.output[0].strip()
    predicted_answer = state.predicted_answer

    question = sample["question_text"]
    rationales = sample["rationale_choice_texts"]

    rationale_prompt = f"""
You are solving the rationale-selection stage of a Visual Commonsense Reasoning (VCR) task.

The model has already selected the following answer:

Selected answer:
{predicted_answer}

Your task is to identify the ONE rationale that best explains why that selected answer is correct.

Reason internally through the following steps:

1. Re-examine the image.
2. Consider the question and the selected answer together.
3. Determine what visual evidence or commonsense connection supports the selected answer.
4. Compare every rationale choice.
5. Reject rationales that:
   - contradict the image,
   - do not explain the selected answer,
   - refer to the wrong person/object/action,
   - introduce unsupported information,
   - or are logically weaker than another choice.
6. Select the rationale that provides the strongest explanation for the selected answer.

Important:
- The rationale must explain the SELECTED ANSWER, not merely describe the image.
- Do not choose a rationale just because it is generally plausible.
- Pay attention to people, actions, objects, and relationships mentioned in the question.
- There is exactly ONE correct rationale.

Question:
{question}

Selected answer:
{predicted_answer}

Rationale choices:

A. {rationales[0]}
B. {rationales[1]}
C. {rationales[2]}
D. {rationales[3]}

After reasoning internally, respond with ONLY the letter of the best rationale:
A, B, C, or D.
"""
    return rationale_prompt
    # print(rationale_prompt)
