import argparse
import os
import random
import sys
from shutil import unpack_archive
from typing import List, Optional
from urllib.request import urlretrieve

import gensim
import numpy as np
from sklearn.metrics import pairwise
from termcolor import colored
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
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


def classify_texts(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    texts: List[str],
    batch_size: int = 32,
    desc: Optional[str] = None,
) -> np.ndarray:
    """
    Computes confidencies for a BERT-based text classifier
    on a list of texts.

    Parameters:
        model: BertForSequenceClassification
        tokenizer: BertTokenizer
        texts: List[str]
        batch_size: int
        desc: str or None

    Returns:
        np.ndarray
    """
    ans = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.no_grad():
            preds = torch.softmax(model(**batch).logits, -1)[:, 1].cpu().numpy()
        ans.append(preds)

    return np.concatenate(ans)


def style_transfer_accuracy(
    preds: List[str], batch_size: int = 32
) -> np.ndarray:
    """
    Computes style transfer accuracies for the list of model predictions.

    Parameters:
        preds: List[str]
        batch_size: int

    Returns:
        np.ndarray
    """
    tokenizer = BertTokenizer.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    )
    model = BertForSequenceClassification.from_pretrained(
        "SkolkovoInstitute/russian_toxicity_classifier"
    ).to(DEVICE)
    ans = classify_texts(
        model,
        tokenizer,
        preds,
        batch_size=batch_size,
        desc="Calculating predictions' toxicity...",
    )
    return ans


def get_sentence_vector(text: str, model, pipeline) -> np.ndarray:
    """
    Computes a vector of a given text

    Parameters:
        text: str
        model: a model used for word embeddings
        pipeline: pipeline used for text preprocessing

    Returns:
        np.ndarray
    """
    processed = pipeline.process(text)
    content = [
        line for line in processed.split("\n") if not line.startswith("#")
    ]
    tagged = [w.split("\t") for w in content if w]
    tokens = []

    for token in tagged:
        if token[3] != "PUNCT":
            tokens.append(token[2])

    embd = [model[token] for token in tokens]
    return np.mean(embd, axis=0).reshape(1, -1)


def cosine_similarity(inputs: List[str], preds: List[str]) -> np.ndarray:
    """
    Computes cosine similarities between vectors of texts' embeddings.

    Parameters:
        inputs: List[str]
        preds: List[str]

    Returns:
        np.ndarray
    """
    print("Loading FastText model...")
    model = gensim.models.KeyedVectors.load(fasttext_filename)
    print("Loading UDPipe model...")
    model_udpipe = Model.load(udpipe_filename)
    pipeline = Pipeline(
        model_udpipe, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
    )
    ans = []

    for text_1, text_2 in tqdm(
        zip(inputs, preds), desc="Calculating cosine similarities..."
    ):
        ans.append(
            pairwise.cosine_similarity(
                get_sentence_vector(text_1, model, pipeline),
                get_sentence_vector(text_2, model, pipeline),
            )
        )

    return np.array(ans)


def fluency_score(
    inputs: List[str], preds: List[str], batch_size: int = 32
) -> np.ndarray:
    """
    Computes fluency scores
    for the two lists of original and predicted sentences.

    Parameters:
        inputs: List[str]
        preds: List[str]
        batch_size: int

    Returns:
        np.ndarray
    """
    tokenizer = BertTokenizer.from_pretrained(
        "SkolkovoInstitute/rubert-base-corruption-detector"
    )
    model = BertForSequenceClassification.from_pretrained(
        "SkolkovoInstitute/rubert-base-corruption-detector"
    ).to(DEVICE)
    input_scores = classify_texts(
        model,
        tokenizer,
        inputs,
        batch_size=batch_size,
        desc="Calculating original sentences' fluency...",
    )
    pred_scores = classify_texts(
        model,
        tokenizer,
        preds,
        batch_size=batch_size,
        desc="Calculating predictions' fluency...",
    )
    ans = pred_scores - input_scores
    ans = ans * 1.15 + 1
    ans = np.maximum(0, ans)
    ans = np.minimum(1, ans)
    return ans


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
    print("Reading sentences...")

    with open(inputs_path) as inputs_file, open(preds_path) as preds_file:
        inputs = inputs_file.readlines()
        preds = preds_file.readlines()

    sta = style_transfer_accuracy(preds, batch_size=batch_size)
    cs = cosine_similarity(inputs, preds)
    fl = fluency_score(inputs, preds, batch_size=batch_size)
    js = sta * cs * fl
    print(f"\nModel name: {model_name}\n")
    print(colored("STA", attrs=["bold"]), f": {np.mean(sta):.2f}")
    print(colored("CS", attrs=["bold"]), f": {np.mean(cs):.2f}")
    print(colored("FL", attrs=["bold"]), f": {np.mean(fl):.2f}")
    print(
        colored("Joint score", attrs=["bold"]),
        ":",
        colored(f": {np.mean(js):.2f}", attrs=["bold"]),
    )

    if results_file:
        if os.path.exists(results_file):
            if os.path.isfile(results_file):
                with open(results_file, "a") as results:
                    results.write(
                        " | ".join([str(sta), str(cs), str(fl), f"**{js}**"])
                    )
            else:
                raise ValueError("Results file is a directory!")
        else:
            with open(results_file, "w") as results:
                results.write("Method | STA↑ | CS↑ | FL↑ | JS↑")
                results.write("------ | ---- | --- | --- | ---")
                results.write(
                    " | ".join([str(sta), str(cs), str(fl), f"**{js}**"])
                )


if __name__ == "__main__":
    fasttext_model_url = "http://vectors.nlpl.eu/repository/20/213.zip"
    fasttext_filename = "ru_fasttext/model.model"

    if not os.path.exists(fasttext_filename):
        print("FastText model not found. Downloading...", file=sys.stderr)
        urlretrieve(
            fasttext_model_url, fasttext_model_url.split("/")[-1], show_progress
        )
        unpack_archive(fasttext_model_url.split("/")[-1], "ru_fasttext")
        os.remove(fasttext_model_url.split("/")[-1])

    udpipe_url = "https://rusvectores.org/static/models/udpipe_syntagrus.model"
    udpipe_filename = udpipe_url.split("/")[-1]

    if not os.path.exists(udpipe_filename):
        print("UDPipe model not found. Downloading...", file=sys.stderr)
        urlretrieve(udpipe_url, udpipe_filename, show_progress)

    parser = argparse.ArgumentParser(
        prog="metric",
        description="Compute metrics for the predictions of a model.",
    )
    parser.add_argument(
        "-i", "--inputs", required=True, help="path to test sentences"
    )
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
