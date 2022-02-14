from typing import List

import numpy as np
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

"""
"""


def style_transfer_accuracy(pred: List[str]) -> float:
    """
    Computes style transfer accuracy for the list of model predictions.

    Parameters:
        pred: List[str]

    Returns:
        float
    """
    print("Calculating style of predictions")
    ans = []

    tokenizer = BertTokenizer.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )
    model = BertForSequenceClassification.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )

    for i in tqdm(0, len(pred), 32):
        batch = tokenizer(pred[i : i + 32], return_tensors="pt", padding=True)
        res = model(**batch)["logits"].argmax(1).float().data.tolist()
        ans.extend([1 - item for item in res])

    return np.mean(ans)


def cosine_similarity():
    """

    Parameters:

    Returns:
    """
    pass


def perplexity():
    """

    Parameters:

    Returns:
    """
    pass


def metric(sta: float, cs: float, ppl: float) -> float:
    """
    Computes the geometric mean between style transfer accuracy,
    cosine similarity and the inverse of perplexity.

    Parameters:
        sta: float
        cs: float
        ppl: float

    Returns:
        float
    """
    return (max(sta, 0.0) * max(cs, 0.0) * max(1.0 / ppl, 0.0)) ** (1 / 3)


def main():
    """

    Parameters:

    Returns:
    """
    pass


if __name__ == "__main__":
    main()
