import os
import torch
import pickle
from utils import make_vocab
from tqdm import tqdm
from typing import List

def read_smiles(smiles_file: str) -> List[str]:
    with open(smiles_file, "r") as f:
        smiles_list = [line.strip().split(" ")[0] for line in f.readlines()]
    return smiles_list

# load smiles
smiles_file = './data/preprocessed/Druglike_million_canonical_no_dot_dup.smi'
out_file = './data/model/moses_vae/vocab.pkl'

smiles_list = read_smiles(smiles_file)
vocab = make_vocab(smiles_list)

# save vocab
with open(out_file, 'wb') as f:
    pickle.dump(vocab, f)

print(f"Vocab saved to {out_file}")

config = {
        "encoder_hidden_size": 256,  # encoderのGRUの隠れ層の次元数h
        "encoder_num_layers": 1,  # encoderのGRUの層数
        "bidirectional": True,  # Trueなら双方向，Falseなら単方向
        "encoder_dropout": 0.5,  # encoderのGRUのdropout率
        "latent_size": 128,  # 潜在変数の次元数z
        "decoder_hidden_size": 512,  # decoderのGRUの隠れ層の次元数h
        "decoder_num_layers": 3,  # decoderのGRUの層数
        "decoder_dropout": 0,  # decoderのGRUのdropout率
        "n_batch": 512,  # バッチサイズ
        "clip_grad": 50,
        "kl_start": 0,
        "kl_w_start": 0,
        "kl_w_end": 0.05,
        "lr_start": 3 * 1e-4,
        "lr_n_period": 10,
        "lr_n_restarts": 10,
        "lr_n_mult": 1,
        "lr_end": 3 * 1e-4,
        "n_last": 1000,
        "n_jobs": 1,
        "n_workers": 1,
        "model_save": None,
        "save_frequency": 10,
    }

out_file = './data/model/moses_vae/config.pkl'

# save config
with open(out_file, 'wb') as f:
    pickle.dump(config, f)

print(f"Config saved to {out_file}")