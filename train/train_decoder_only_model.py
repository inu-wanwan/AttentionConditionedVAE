import sys
import os
import wandb
import pickle
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from scripts.utils import load_config, make_vocab
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor
from src.generation_models.moses_vae import SmilesVAE
from src.generation_models.decoder_only_model import DecoderOnlyCVAE
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

class Trainer:
    def __init__(self, model, optimizer, current_time, device):
        self.model = model
        self.optimizer = optimizer
        self.current_time = current_time
        self.device = device

    def _epoch_step(self, pbar, train):
        """
        train が True のとき学習を行い、False のとき検証を行う
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        recon_loss = 0.0
        for batch in pbar:
            # データを device に転送して定義
            smiles_embedding = batch['smiles_embedding'].to(self.device)
            af2_embedding = batch['protein_embedding'].to(self.device)

            # padding された smiles tensor はデータセットに含まれないのでここで tensorize と padding を実装する
            smiles_tensor = [self.model.pretrained_smiles_vae.string2tensor(smiles) for smiles in batch['smiles']]
            smiles_padded_tensor = nn.utils.rnn.pad_sequence(
                smiles_tensor, batch_first=True, padding_value=self.model.pretrained_smiles_vae.PAD
            ).to(self.device)

            # forward pass
            loss = self.model(smiles_padded_tensor, smiles_embedding, af2_embedding)

            if train:
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # log
            recon_loss += loss.item()
            pbar.set_postfix({'loss': loss.item() / len(batch)})
            if train:
                wandb.log({'train_batch_loss': loss.item() / len(batch)})
            else:
                wandb.log({'val_batch_loss': loss.item() / len(batch)})

        return recon_loss
    
    def train(self, epochs, train_dataloader, val_dataloader, checkpoint_dir, save_freq):
        for epoch in range(epochs):
            # train
            print(f"Epoch {epoch + 1} / {epochs}")
            train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} / {epochs} Train")
            train_loss = self._epoch_step(train_pbar, train=True)

            # validation
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} / {epochs} Val")
            val_loss = self._epoch_step(val_pbar, train=False)

            print(f"Epoch {epoch + 1} / {epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # wandb log
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_dataloader),
                'val_loss': val_loss / len(val_dataloader)
                })
            
            # save the model per epoch
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                wandb.save(f"checkpoint/model_epoch_{epoch}.pth")

        # save the model
        model_save_path = os.path.join(checkpoint_dir, f"model.pth")
        torch.save(self.model.state_dict(), model_save_path)
        wandb.save(f"checkpoint/model.pth")

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"GPU ID: {args.gpu}")

    # load config
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')['fnta']
    model_config = load_config(os.path.join(args.config_dir, args.model_config))
    train_config = load_config(os.path.join(args.config_dir, args.train_config))

    # set target
    target = model_config['target']

    # override config
    if args.epochs is not None:
        train_config['epochs'] = args.epochs
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
    if args.lr is not None:
        train_config['lr'] = args.lr

    # initialize WandB
    wandb.init(
        project='Attention Conditioned VAE (decoder only model)', 
        config={"train_config": train_config, "model_config": model_config},
        name=f"{target}_decoder_only_model"
        )

    # set current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # directories
    os.makedirs(os.path.join(file_config['data']['model'], 'decoder_only_model'), exist_ok=True)
    os.makedirs(os.path.join(file_config['data']['model'], 'decoder_only_model', f"{target}_{current_time}"), exist_ok=True)
    model_save_dir = os.path.join(file_config['data']['model'], 'decoder_only_model', f"{target}_{current_time}")

    # save config files
    config_save_path = os.path.join(model_save_dir, 'model_config.pkl')
    with open(config_save_path, 'wb') as f:
        pickle.dump(model_config, f)
    print(f"Model config saved at {config_save_path}")

    # data files
    train_file = os.path.join(file_config['data']['train'], model_config['train_file'])
    val_file = os.path.join(file_config['data']['val'], model_config['val_file'])

    # train parameters
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    lr = float(train_config['lr'])
    save_freq = train_config['save_freq']

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
        batch_size=16,
        shuffle=False,
    )

    # model

    # vae pretrained model
    # model file 
    vae_model_file = os.path.join(file_config['data']['moses'], model_config['moses_vae_file'])
    vocab_file = os.path.join(file_config['data']['moses'], 'vocab.pkl')
    config_file = os.path.join(file_config['data']['moses'], 'config.pkl')
    
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

    # docking score prediction model
    # model file 
    docking_model_file = os.path.join(file_config['data']['docking'], model_config['docking_score_regression'], 'model.pth')
    docking_model_config = os.path.join(file_config['data']['docking'], model_config['docking_score_regression'], 'model_config.pkl')
    
    try:
        with open(docking_model_config, 'rb') as f:
            docking_model_config = pickle.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {docking_model_config}")

    if docking_model_config['target'] != target:
        print(f"Target mismatch: {docking_model_config['target']} != {target}")
        sys.exit(1)
 
    # docking model load
    docking_model = DockingScorePredictor(
        embed_dim=docking_model_config['embed_dim'],
        num_heads=docking_model_config['num_heads'],
        ffn_hidden_dim=docking_model_config['ffn_hidden_dim'],
        num_transformer_blocks=docking_model_config['num_transformer_blocks'],
        return_attn_wts=True,
    ).to(torch.device('cuda'))

    docking_model.load_state_dict(torch.load(docking_model_file))

    # model
    model = DecoderOnlyCVAE(
        smiles_vae=vae_model,
        docking_score_predictor=docking_model,
        af2_max_len=model_config['protein_max_len'],
        transformer_layer_used=model_config['transformer_layer_used'],
    ).to(torch.device('cuda'))

    # training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, current_time, torch.device('cuda'))

    trainer.train(epochs, train_dataloader, val_dataloader, model_save_dir, save_freq)
    