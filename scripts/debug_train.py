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
    model_config = load_config('model.yml')['sample_model']
    train_config = load_config('train.yml')['sample_model']

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

    # model
    model = DockingScorePredictor(
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        ffn_hidden_dim=model_config['ffn_hidden_dim'],
        num_transformer_blocks=1,
    ).cuda()

    # loss function
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    train_loss = 0.0
    smiles_embedding = torch.randn(1, 100, 384).cuda()
    af2_embedding = torch.randn(1, 1390, 384).cuda()
    smiles_mask = torch.ones(1, 100).cuda()
    af2_mask = torch.ones(1, 1390).cuda()

    score_pred = model(smiles_embedding, af2_embedding, smiles_mask, af2_mask)

    image = make_dot(score_pred, params=dict(model.named_parameters()))
    image.format = 'png'
    image.render("DockingScorePredictor")



if __name__ == '__main__':
    debug()