import sys
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import load_config
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor

# load config
file_config = load_config('filepath.yml')
data_config = load_config('data.yml')
model_config = load_config('model.yml')['generation_model']
train_config = load_config('train.yml')['generation_train']

# directories
alphafold_dir = file_config['data']['alphafold']
data_dir = file_config['data']['samples']
model_save_dir = file_config['model']

# data files
train_file = os.path.join(file_config['data']['train'], 'train.csv')
val_file = os.path.join(file_config['data']['val'], 'val.csv')
test_file = os.path.join(file_config['data']['test'], 'test.csv')

# train parameters
batch_size = train_config['batch_size']
epochs = train_config['epochs']
lr = float(train_config['lr'])

# dataloaders
train_dataloader = get_dataloader(
    csv_file=train_file,
    smiles_max_len=data_config['dataset']['smiles_max_len'],
    protein_max_len=data_config['dataset']['protein_max_len'],
    batch_size=batch_size,
    shuffle=True,
)

val_dataloader = get_dataloader(
    csv_file=val_file,
    smiles_max_len=data_config['dataset']['smiles_max_len'],
    protein_max_len=data_config['dataset']['protein_max_len'],
    batch_size=batch_size,
    shuffle=False,
)

# model
model = DockingScorePredictor(
    embed_dim=model_config['embed_dim'],
    num_heads=model_config['num_heads'],
    ffn_hidden_dim=model_config['ffn_hidden_dim'],
    num_transformer_blocks=model_config['num_transformer_layers'],
    return_attn_wts=True
)


for batch_idx, batch in enumerate(train_dataloader):
    print(f"Batch {batch_idx}")
    print(f"    SMILES: {len(batch['smiles'])}")
    print(f"    SMILES: {batch['smiles_embedding'].shape}")
    # print(f"    SMILES mask: {batch['smiles_mask'].shape}")
    # print(f"    SMILES mask: {batch['smiles_mask'][0]}")
    # print(f"    Protein embedding: {batch['protein_embedding'].shape}")
    # print(f"    Protein mask: {batch['protein_mask'].shape}")
    # print(f"    Protein mask; {batch['protein_mask'][0]}")
    # print(f"    Docking score: {batch['docking_score']}")
    model.eval()
    with torch.no_grad():
        docking_score, self_attn_wts_list, cross_attn_wts_list = model(batch['smiles_embedding'], batch['protein_embedding'])
        print(f"    Docking score: {docking_score}")
        print(f"    Self attention weights: {len(self_attn_wts_list)}")
        print(f"    Cross attention weights: {len(cross_attn_wts_list)}")
        print(f"    Self attention weights: {self_attn_wts_list[0].shape}")
        print(f"    Cross attention weights: {cross_attn_wts_list[0].shape}")
    break