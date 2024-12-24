# utility functions for the project
import os
import yaml
import csv
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
