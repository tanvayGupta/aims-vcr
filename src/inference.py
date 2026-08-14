"""
inference.py
-------------
The function that actually calls the model and scores its reply. Mirrors
the notebook's "Run inference" section.
"""

import config
import state


def output_generator(model, processor, prompt, answer_or_rationale, sample):
    """Run one generation pass: send `state.image` + `prompt` to the model,
    decode a single-letter reply, print it next to the ground-truth letter,
    and tally it into state.correct_answers or state.correct_rationales.

    answer_or_rationale: True means this call is scoring the Stage 1 answer
    prompt; False means it's scoring the Stage 2 rationale prompt.
    """

    # Qwen2.5-VL's expected chat format: one user turn holding an image
    # block and a text block.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": state.image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # turn the structured `messages` into the exact prompt string the model
    # was trained on, adding the special tokens that tell it "now it's your
    # turn to reply" (add_generation_prompt=True)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # tokenize the text and preprocess the image together into model-ready
    # tensors, then move them to the model's device
    inputs = processor(
        text=[text],
        images=[state.image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # only 1 new token is generated since the whole reply is a single
    # letter (A-D)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,  # Limits to one token
        do_sample=config.DO_SAMPLE,
        # do_sample=False is greedy decoding, which is mathematically the
        # same as sampling at temperature -> 0: always take the argmax token
    )

    # model.generate() returns the prompt tokens *and* the new token
    # concatenated together, so record how many tokens were prompt...
    input_length = inputs["input_ids"].shape[1]
    # ...and slice those off, keeping only the newly generated token(s)
    generated_ids = generated_ids[:, input_length:]

    # convert the generated token id(s) back into a plain string
    # (skip_special_tokens drops things like the EOS token)
    state.output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # --- Stage 1 (answer) branch ---
    if answer_or_rationale:
        print("Model:", state.output[0])
        # sample["answer_label"] is the correct choice's index (0-3);
        # "ABCD"[index] turns it back into a letter
        print("Correct:", "ABCD"[sample["answer_label"]])
        if state.output[0].strip() == "ABCD"[sample["answer_label"]]:
            state.correct_answers += 1

    # --- Stage 2 (rationale) branch, mirrors the block above using
    # rationale_label ---
    elif not answer_or_rationale:
        print("Model rationale:", state.output[0])
        print("Correct rationale:", "ABCD"[sample["rationale_label"]])

        if state.output[0].strip() == "ABCD"[sample["rationale_label"]]:
            state.correct_rationales += 1
