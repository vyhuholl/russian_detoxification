import argparse
import os
import sys
from typing import List
from urllib.request import urlretrieve

from tqdm import tqdm
from ufal.udpipe import Model, Pipeline


def tokenize(
    text: str, pipeline: Pipeline, tags: bool = False, lemmas: bool = False
) -> List[str]:
    """
    Tokenizes a text with the UDPipe pipeline.

    Parameters:
        inputs_path: str
        preds_path: str
        results_file: str
    Returns:
        None
    """
    processed = pipeline.process(text)
    content = [
        line for line in processed.split("\n") if not line.startswith("#")
    ]
    tagged = [w.split("\t") for w in content if w]
    tokens = []

    for token in tagged:
        if token[3] == "PUNCT":
            continue
        token_res = ""
        if lemmas:
            token_res = token[2]
        else:
            token_res = token[1]
        if tags:
            token_res += "_" + token[3]
        tokens.append(token_res)

    return tokens


def main(inputs_path: str, vocab_path: str, results_path: str) -> None:
    """
    Deletes all rude or toxic words (including lemmas)
    from a pre-defined vocab in a list of texts,
    and writes the resulting texts to a file.

    Parameters:
        inputs_path: str
        preds_path: str
        results_path: str

    Returns:
        None
    """
    udpipe_url = "https://rusvectores.org/static/models/udpipe_syntagrus.model"
    udpipe_filename = udpipe_url.split("/")[-1]

    if not os.path.exists(udpipe_filename):
        print("UDPipe model not found. Downloading...", file=sys.stderr)
        urlretrieve(udpipe_url, udpipe_filename)

    print("Loading UDPipe model...")
    model_udpipe = Model.load(udpipe_filename)
    pipeline = Pipeline(
        model_udpipe, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
    )

    with open(inputs_path) as inputs_file:
        texts = inputs_file.readlines()

    with open(vocab_path) as vocab_file:
        vocab = [line.strip() for line in vocab_file.readlines()]

    results = []

    for text in tqdm(texts, desc="Deleting toxic words..."):
        words = tokenize(text, pipeline, lemmas=False)
        lemmas = tokenize(text, pipeline, lemmas=True)
        clean_text = " ".join(
            [word for word, lemma in zip(words, lemmas) if lemma not in vocab]
        )
        results.append(clean_text)

    with open(results_path, "w") as results_file:
        for text in results:
            results_file.write(text + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="delete",
        description="Deletes all rude or toxic words from a pre-defined "
        + "vocab in a list of texts, "
        + "and writes the resulting texts to a file.",
    )
    parser.add_argument(
        "-i", "--inputs", required=True, help="path to test sentences"
    )
    parser.add_argument(
        "-v", "--vocab", required=True, help="path to the vocab"
    )
    parser.add_argument(
        "-f", "--file", required=True, help="path to results file"
    )
    args = parser.parse_args()
    main(args.inputs, args.vocab, args.file)
