import sys
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.utils import load_config
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor
from datetime import datetime

def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(tensor)
        raise ValueError(f"NaN found in {name}")

def train():
    """
    Train the docking score prediction model.
    """
    # load config
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')
    model_config = load_config('model.yml')['docking_score_regression_model']
    train_config = load_config('train.yml')['docking_score_regression_train']

    # initialize WandB
    wandb.init(project='Docking score regression transformer', config={"train_config": train_config, "model_config": model_config})

    # set current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # directories
    alphafold_dir = file_config['data']['alphafold']
    data_dir = file_config['data']['samples']
    model_dir = os.path.join(file_config['data']['model'], 'ds_regression')

    os.makedirs(os.path.join(model_dir, f"ds_{current_time}"), exist_ok=True)
    model_save_dir = os.path.join(model_save_dir, f"ds_{current_time}")

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
        num_transformer_blocks=model_config['num_transformer_blocks'],
    ).cuda()

    # loss function
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} / {epochs} Train")
        for _, batch in enumerate(train_pbar):
            smiles_embedding = batch['smiles_embedding'].cuda()
            af2_embedding = batch['protein_embedding'].cuda()
            docking_score = batch['docking_score'].cuda()

            # forward pass
            optimizer.zero_grad()
            docking_score_pred = model(smiles_embedding, af2_embedding)
            loss = criterion(docking_score_pred.squeeze(), docking_score)

            # back propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item() / batch_size})
            wandb.log({'train_batch_loss': loss.item() / batch_size}) 

        wandb.log({'train_loss': train_loss / len(train_dataloader)})

        # validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} / {epochs} Val")
        for _, batch in enumerate(val_pbar):
            with torch.no_grad():
                smiles_embedding = batch['smiles_embedding'].cuda()
                af2_embedding = batch['protein_embedding'].cuda()
                docking_score = batch['docking_score'].cuda()

                docking_score_pred = model(smiles_embedding, af2_embedding)
                loss = criterion(docking_score_pred.squeeze(), docking_score)

                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item() / batch_size})
                wandb.log({'val_batch_loss': loss.item() / batch_size})
        val_loss /= len(val_dataloader)
        wandb.log({'val_loss': val_loss})
        print(f"Epoch {epoch + 1} / {epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # save the model per epoch
        checkpoint_path = os.path.join(model_save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(f"checkpoint/model_epoch_{epoch}.pth")

    # save the model
    model_save_path = os.path.join(model_save_dir, f"model.pth")
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == '__main__':
    train()
