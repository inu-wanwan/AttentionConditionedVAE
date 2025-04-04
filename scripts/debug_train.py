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
from torchviz import make_dot


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(tensor)
        raise ValueError(f"NaN found in {name}")

def debug():
    """
    Train the docking score prediction model.
    """
    # load config
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')
    model_config = load_config('model.yml')['generation_model']
    train_config = load_config('train.yml')['generation_train']

    # initialize WandB
    # wandb.init(project='sample_model', config=train_config)

    # directories
    alphafold_dir = file_config['data']['alphafold']
    data_dir = file_config['data']['samples']

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
    
    pbar = tqdm(train_dataloader, desc="Training")
    print(f"len(train_dataloader): {len(pbar)}")


if __name__ == '__main__':
    debug()