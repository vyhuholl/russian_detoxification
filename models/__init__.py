import os
import sys
import pickle
import re
from shutil import unpack_archive
from urllib.request import urlretrieve

import gensim
import numpy as np
import torch
from ..metric import get_sentence_vector
from model_classes import MODEL_CLASSES
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from ufal.udpipe import Model, Pipeline
from ..utils import show_progress

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

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


class Detoxifier:
    def __init__(self, model_class: str, clf_path: str):
        self.name = MODEL_CLASSES[model_class][2]
        self.model = MODEL_CLASSES[model_class][0].from_pretrained(self.name)
        self.model.to(DEVICE)
        self.tokenizer = MODEL_CLASSES[model_class][1].from_pretrained(
            self.name
        )

        with open(clf_path, "rb") as clf_file:
            self.clf = pickle.load(clf_file)
