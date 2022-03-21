import argparse
import re
from typing import List

from pymorphy2 import MorphAnalyzer
from tqdm import tqdm


def delete_toxic_words(
    text: List[str], morph: MorphAnalyzer, vocab: List[str], lemmas: List[str]
) -> str:
    """
    Deletes from a list of words all words which are either in toxic vocab
    or have a lemma that is in the list of toxic words' lemmas.
    Returns a single string — the remaining words joined by space.

    Parameters:
        text: List[str]
        morph: MorphAnalyzer
        vocab: List[str]
        lemmas: List[str]

    Returns:
        str
    """
    return " ".join(
        [
            word
            for word in text
            if word not in vocab
            and morph.parse(word)[0].normal_form not in lemmas
        ]
    )


def main(inputs_path: str, vocab_path: str, results_file: str) -> None:
    """
    Deletes all rude or toxic words (including lemmas)
    from a pre-defined vocab in a list of texts,
    and writes the resulting texts to a file.

    Parameters:
        inputs_path: str
        preds_path: str
        results_file: str
    Returns:
        None
    """
    with open(inputs_path) as inputs_file:
        texts = inputs_file.readlines()

    with open(vocab_path) as vocab_file:
        vocab = vocab_file.readlines()

    morph = MorphAnalyzer()
    vocab = [word for word in vocab if " " not in word]
    print("Lemmatizing vocab...")
    lemmas = [morph.parse(word)[0].normal_form for word in tqdm(vocab)]
    clean_texts = []
    print("Preprocessing and tokenizing texts...")

    for text in tqdm(texts):
        clean_texts.append(
            [word for word in re.split(r"[^а-яё]+", text.lower()) if word]
        )

    print("Deleting rude and toxic words...")
    texts = [
        delete_toxic_words(text, morph, vocab, lemmas) for text in tqdm(texts)
    ]

    with open(results_file, "w") as results:
        results.writelines(texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="delete",
        description="Deletes all rude or toxic words from a pre-defined"
        + "vocab in a list of texts,"
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
