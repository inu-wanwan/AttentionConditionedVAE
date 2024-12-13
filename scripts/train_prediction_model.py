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

def train():
    """
    Train the docking score prediction model.
    """
    # load config
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')
    model_config = load_config('model.yml')['sample_model']
    train_config = load_config('train.yml')['sample_model']

    # initialize WandB
    wandb.init(project='sample_model', config=train_config)

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
        num_transformer_blocks=model_config['num_transformer_layers'],
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
            smiles_mask = batch['smiles_mask'].cuda()
            af2_mask = batch['protein_mask'].cuda()
            docking_score = batch['docking_score'].cuda()

            # forward pass
            optimizer.zero_grad()
            docking_score_pred = model(smiles_embedding, af2_embedding, smiles_mask, af2_mask)
            loss = criterion(docking_score_pred.squeeze(), docking_score)

            # back propagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        # wandb logging per epoch
        train_loss /= len(train_dataloader)
        wandb.log({'train_loss': train_loss})

        # validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} / {epochs} Val")
        with torch.no_grad():
            for smiles_embedding, af2_embedding, smiles_mask, af2_mask, docking_score in val_pbar:
                smiles_embedding = smiles_embedding.cuda()
                af2_embedding = af2_embedding.cuda()
                smiles_mask = smiles_mask.cuda()
                af2_mask = af2_mask.cuda()
                docking_score = docking_score.cuda()

                docking_score_pred = model(smiles_embedding, af2_embedding, smiles_mask, af2_mask)
                loss = criterion(docking_score_pred.squeeze(), docking_score)

                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})

        val_loss /= len(val_dataloader)
        wandb.log({'val_loss': val_loss})
        print(f"Epoch {epoch + 1} / {epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # save the model
    torch.save(model.state_dict(), "docking_score_predictor.pth")
    wandb.save("docking_score_predictor.pth")
    print("Model saved to docking_score_predictor.pth")

if __name__ == '__main__':
    train()