import numpy as np

"""
"""


def style_transfer_accuracy():
    """

    Parameters:

    Returns:
    """
    pass


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


def metric(sta: np.float32, cs: np.float32, ppl: np.float32) -> np.float32:
    """
    Computes the geometric mean between style transfer accuracy,
    cosine similarity and the inverse of perplexity.

    Parameters:
        sta: np.float32
        cs: np.float32
        ppl: np.float32

    Returns:
        np.float32
    """
    return (max(sta, 0.0) * max(cs, 0.0) * max(1 / ppl, 0.0)) ** (1 / 3)


def main():
    """

    Parameters:

    Returns:
    """
    pass


if __name__ == "__main__":
    main()
