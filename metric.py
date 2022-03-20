import argparse
import os
import random
import sys
from shutil import unpack_archive
from typing import List, Optional
from urllib.request import urlretrieve

import gensim
import numpy as np
import torch
from sklearn.metrics import pairwise
from termcolor import colored
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from ufal.udpipe import Model, Pipeline
from utils import show_progress

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

pbar = None

fasttext_model_url = "http://vectors.nlpl.eu/repository/20/213.zip"
fasttext_filename = "ru_fasttext/model.model"

if not os.path.exists(fasttext_filename):
    print("FastText model not found. Downloading...", file=sys.stderr)
    urlretrieve(fasttext_model_url, fasttext_model_url.split("/")[-1], show_progress)
    unpack_archive(fasttext_model_url.split("/")[-1], "ru_fasttext")
    os.remove(fasttext_model_url.split("/")[-1])

model = gensim.models.KeyedVectors.load(fasttext_filename)

udpipe_url = "https://rusvectores.org/static/models/udpipe_syntagrus.model"
udpipe_filename = udpipe_url.split("/")[-1]

if not os.path.exists(udpipe_filename):
    print("UDPipe model not found. Downloading...", file=sys.stderr)
    urlretrieve(udpipe_url, udpipe_filename, show_progress)

model_udpipe = Model.load(udpipe_filename)
process_pipeline = Pipeline(
    model_udpipe, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
)


def style_transfer_accuracy(preds: List[str], batch_size: int = 32) -> float:
    """
    Computes style transfer accuracy for the list of model predictions.

    Parameters:
        preds: List[str]
        batch_size: int

    Returns:
        float
    """
    print("Calculating style of predictions...")
    ans = []

    tokenizer = BertTokenizer.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )
    model = BertForSequenceClassification.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )

    for i in tqdm(0, len(preds), batch_size):
        batch = tokenizer(preds[i : i + batch_size], return_tensors="pt", padding=True)
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
    print("Calculating cosine similarities...")
    ans = []

    for text_1, text_2 in tqdm(zip(inputs, preds)):
        ans.append(
            pairwise.cosine_similarity(
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
    print("Calculating perplexity...")
    lls = []
    weights = []

    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt2large")
    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt2large")
    model.to(DEVICE)

    for text in tqdm(preds):
        encodings = tokenizer(f"\n{text}\n", return_tensors="pt")
        input_ids = encodings.input_ids.to(DEVICE)
        target_ids = input_ids.clone()
        w = max(0, len(input_ids[0]) - 1)
        if w > 0:
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs[0]
                ll = log_likelihood.item()
        else:
            ll = 0
        lls.append(ll)
        weights.append(w)

    return np.dot(lls, weights) / sum(weights)


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


def main(
    inputs_path: str,
    preds_path: str,
    batch_size: int,
    model_name: str,
    results_file: Optional[str],
) -> None:
    """
    Computes metrics and writes them to console and to the results file.

    Parameters:
        inputs_path: str
        preds_path: str
        batch_size: int
        model_name: str
        results_file: str or None
    Returns:
        None
    """
    with open(inputs_path) as inputs_file, open(preds_path) as preds_file:
        inputs = inputs_file.readlines()
        preds = preds_file.readlines()

    sta = style_transfer_accuracy(preds, batch_size=batch_size)
    cs = cosine_similarity(inputs, preds)
    ppl = perplexity(preds)
    gm = metric(sta, cs, ppl)
    print(f"Model name: {model_name}\n")
    print(colored("STA", attrs=["bold"]), f": {sta:.2f}")
    print(colored("CS", attrs=["bold"]), f": {cs:.2f}")
    print(colored("PPL", attrs=["bold"]), f": {ppl:.2f}")
    print(
        colored("Geometric mean", attrs=["bold"]),
        ":",
        colored(f": {sta:.2f}", attrs=["bold"]),
    )

    if results_file:
        if os.path.exists(results_file):
            if os.path.isfile(results_file):
                with open(results_file, "a") as results:
                    results.write(
                        " | ".join([str(sta), str(cs), str(ppl), f"**{gm}**"])
                    )
            else:
                raise ValueError("Results file is a directory!")
        else:
            with open(results_file, "w") as results:
                results.write("Method | STA↑ | CS↑ | PPL↓ | GM↑")
                results.write("------ | ---- | --- | ----| ---")
                results.write(" | ".join([str(sta), str(cs), str(ppl), f"**{gm}**"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="metric", description="Compute metrics for the predictions of a model."
    )
    parser.add_argument("-i", "--inputs", required=True, help="path to test sentences")
    parser.add_argument(
        "-p", "--preds", required=True, help="path to predictions of a model"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        help="batch size for the toxicity classifier",
    )
    parser.add_argument("-m", "--model", default="", help="model name")
    parser.add_argument("-f", "--file", default=None, help="results file")
    args = parser.parse_args()
    main(args.inputs, args.preds, args.batch_size, args.model, args.file)
