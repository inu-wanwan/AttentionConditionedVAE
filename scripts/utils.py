# utility functions for the project
import os
import yaml
import csv
import numpy as np
import pickle
import torch
import pandas as pd
from typing import List
from src.generation_models.moses_vae import SmilesVAE
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor

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
    
# 事前学習済み vae model を読み込む
def load_vae_model() -> SmilesVAE:
    path_config = load_config('filepath.yml')
    # model file 
    vae_model_file = os.path.join(path_config['data']['moses'], 'smiles_vae_dmqp1m_no_dot_dup.pt')
    vocab_file = os.path.join(path_config['data']['moses'], 'vocab.pkl')
    config_file = os.path.join(path_config['data']['moses'], 'config.pkl')
    
    # vocab for vae model
    try:
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print(f"Vocab file not found: {vocab_file}")

    # config for vae model
    try:
        with open(config_file, 'rb') as f:
            moses_config = pickle.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_file}")
    
    # vae model load 
    vae_model = SmilesVAE(
        device=torch.device('cuda'),
        vocab=vocab,
        config=moses_config
    ).to(torch.device('cuda'))

    vae_model.load_state_dict(torch.load(vae_model_file))

    return vae_model

# timestamp から docking score prediction model を読み込む
def load_docking_score_predictor(timestamp: str) -> DockingScorePredictor:
    path_config = load_config('filepath.yml')
    model_config = load_config('model.yml')['docking_score_regression_model']
    model_dir = path_config['data']['docking']
    model_path = os.path.join(model_dir, f"ds_{timestamp}", "model.pth")
    
    # model
    model = DockingScorePredictor(
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        ffn_hidden_dim=model_config['ffn_hidden_dim'],
        num_transformer_blocks=model_config['num_transformer_blocks'],
        return_attn_wts=True
    )

    model.load_state_dict(torch.load(model_path))

    return model