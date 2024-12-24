# utility functions for the project
import os
import yaml
import csv
import numpy as np
import pandas as pd
from typing import List

def load_config(config_file):
    config_path = os.path.join('config', config_file)
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config

# smilesファイルを読み込む
def read_smiles(smiles_file: str) -> List[str]:
    df = pd.read_csv(smiles_file)
    return df["canonical_smiles"].tolist()

def make_vocab(smiles_list: List[str]) -> dict:
    vocab = {}
    for smiles in smiles_list:
        for c in smiles:
            if c not in vocab:
                vocab[c] = len(vocab)
    vocab["PAD"] = len(vocab)
    vocab["SOS"] = len(vocab)
    vocab["EOS"] = len(vocab)
    vocab["UNK"] = len(vocab)
    return vocab

class CircularBuffer:
    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]

    def mean(self):
        if self.size > 0:
            return self.data[: self.size].mean()
        return 0.0