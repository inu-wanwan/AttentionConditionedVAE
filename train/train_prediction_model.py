import sys
import os
import wandb
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.utils import load_config
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', '-c', type=str, default='config/', help='Directory containing config files')
    parser.add_argument('--train_config', '-t', type=str, default='train.yml', help='Training config file')
    parser.add_argument('--model_config', '-m', type=str, default='model.yml', help='Model config file')
    parser.add_argument('--epochs', '-e', type=int, default=None, help='Override the number of epochs in the config file')
    parser.add_argument('--batch_size', '-b', type=int, default=None, help='Override the batch size in the config file')
    parser.add_argument('--lr', '-l', type=float, default=None, help='Override the learning rate in the config file')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()

def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(tensor)
        raise ValueError(f"NaN found in {name}")

def train(args):
    """
    Train the docking score prediction model.
    """
    # set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Using GPU {args.gpu}")
    
    # load config
    file_config = load_config('filepath.yml')
    model_config = load_config(os.path.join(args.config_dir, args.model_config))
    train_config = load_config(os.path.join(args.config_dir, args.train_config))

    # Override parameters if specified in arguments
    if args.epochs is not None:
        train_config['epochs'] = args.epochs
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
    if args.lr is not None:
        train_config['lr'] = args.lr

    # initialize WandB
    wandb.init(
        project='(local) Docking score regression transformer', 
        group=model_config['target'], 
        name=f"target_{model_config['target']}_batch_{train_config['batch_size']}_lr_{train_config['lr']}",
        config={"train_config": train_config, "model_config": model_config}
        )

    # set current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # directories
    alphafold_dir = file_config['data']['alphafold']
    data_dir = file_config['data']['samples']
    model_dir = os.path.join(file_config['data']['model'], 'ds_regression')

    os.makedirs(os.path.join(model_dir, f"ds_{current_time}"), exist_ok=True)
    model_save_dir = os.path.join(model_dir, f"ds_{current_time}")
    
    # save config files
    config_save_path = os.path.join(model_save_dir, 'model_config.pkl')
    with open(config_save_path, 'wb') as f:
        pickle.dump(model_config, f)
    print(f"Model config saved at {config_save_path}")


    # data files
    train_file = os.path.join(file_config['data']['train'], train_config['train_file'])
    val_file = os.path.join(file_config['data']['val'], train_config['val_file'])

    # train parameters
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    lr = float(train_config['lr'])

    # dataloaders
    train_dataloader = get_dataloader(
        csv_file=train_file,
        smiles_max_len=model_config['smiles_max_len'],
        protein_max_len=model_config['protein_max_len'],
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = get_dataloader(
        csv_file=val_file,
        smiles_max_len=model_config['smiles_max_len'],
        protein_max_len=model_config['protein_max_len'],
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
    args = parse_args()
    train(args)
