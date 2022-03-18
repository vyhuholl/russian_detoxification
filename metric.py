import os
import shutil
import sys
from typing import List

import gensim
import numpy as np
import torch
import wget
from sklearn import metrics
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from ufal.udpipe import Model, Pipeline

DATA_DIR = "/data"

fasttext_model_url = "http://vectors.nlpl.eu/repository/20/213.zip"
fasttext_filename = "ru_fasttext/model.model"

if not os.path.isfile(os.path.join(DATA_DIR, fasttext_filename)):
    print("FastText model not found. Downloading...", file=sys.stderr)
    wget.download(fasttext_model_url, out=DATA_DIR)
    shutil.unpack_archive(
        os.path.join(DATA_DIR, fasttext_model_url.split("/")[-1]),
        os.path.join(DATA_DIR, "ru_fasttext/"),
    )

model = gensim.models.KeyedVectors.load(os.path.join(DATA_DIR, fasttext_filename))

udpipe_url = "https://rusvectores.org/static/models/udpipe_syntagrus.model"
udpipe_filename = udpipe_url.split("/")[-1]

if not os.path.isfile(os.path.join(DATA_DIR, udpipe_filename)):
    print("\nUDPipe model not found. Downloading...", file=sys.stderr)
    wget.download(udpipe_url, out=DATA_DIR)

model_udpipe = Model.load(os.path.join(DATA_DIR, udpipe_filename))
process_pipeline = Pipeline(
    model_udpipe, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
)


def style_transfer_accuracy(preds: List[str]) -> float:
    """
    Computes style transfer accuracy for the list of model predictions.

    Parameters:
        preds: List[str]

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

    for i in tqdm(0, len(preds), 32):
        batch = tokenizer(preds[i : i + 32], return_tensors="pt", padding=True)
        res = model(**batch)["logits"].argmax(1).float().data.tolist()
        ans.extend([1 - item for item in res])

    return np.mean(ans)


def get_sentence_vector(text: str) -> np.ndarray:
    """
    Computes a vector of a given text

    Parameters:
        text: str

    Returns:
        np.ndarray
    """
    processed = process_pipeline.process(text)
    content = [line for line in processed.split("\n") if not line.startswith("#")]
    tagged = [w.split("\t") for w in content if w]

    tokens = []

    for token in tagged:
        if token[3] != "PUNCT":
            tokens.append(token[2])

    embd = [model[token] for token in tokens]

    return np.mean(embd, axis=0).reshape(1, -1)


def cosine_similarity(inputs: List[str], preds: List[str]) -> float:
    """
    Computes cosine similarity between vectors of texts' embeddings.

    Parameters:
        inputs: List[str]
        preds: List[str]

    Returns:
        float
    """
    print("Calculating cosine similarities")
    ans = []

    for text_1, text_2 in tqdm(zip(inputs, preds)):
        ans.append(
            metrics.pairwise.cosine_similarity(
                get_sentence_vector(text_1), get_sentence_vector(text_2)
            )
        )

    return np.mean(ans)


def perplexity(preds: List[str]) -> float:
    """
    Computes the perplexity for the list of sentences.

    Parameters:
        preds: List[str]

    Returns:
        float
    """
    print("Calculating perplexity")

    for text in tqdm(preds):
        pass

    return 0.0


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
