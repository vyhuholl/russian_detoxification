import os
import sys
import pickle
import re
from shutil import unpack_archive
from urllib.request import urlretrieve

import numpy as np
import torch
from model_classes import MODEL_CLASSES
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from ..utils import show_progress

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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

    def detoxify(self, text: str) -> str:
        """
        Detoxifies text.

        Parameters:
            text: str

        Returns:
            str
        """
        words = [word for word in re.split(r"[^а-яё]+", text.lower()) if word]
        clean_text = ""
        return clean_text
