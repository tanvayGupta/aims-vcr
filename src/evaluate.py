def label_to_letter(label):
    return "ABCD"[label]


def answer_correct(prediction, sample):
    ground_truth = label_to_letter(sample["answer_label"])
    return prediction == ground_truth


def rationale_correct(prediction, sample):
    ground_truth = label_to_letter(sample["rationale_label"])
    return prediction == ground_truth