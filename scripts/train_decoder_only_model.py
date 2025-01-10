import sys
import os
import wandb
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import load_config, make_vocab
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor
from src.generation_models.moses_vae import SmilesVAE
from src.generation_models.decoder_only_model import DecoderOnlyCVAE
from datetime import datetime

class Trainer:
    def __init__(self, model, optimizer, current_time, device):
        self.model = model
        self.optimizer = optimizer
        self.current_time = current_time
        self.device = device

    def _epoch_step(self, pbar, train):
        """
        Optimizer が与えられた場合は学習を行い、与えられない場合は validation を行う
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
    
    def train(self, epochs, train_dataloader, val_dataloader, checkpoint_dir):
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
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

if __name__ == '__main__':
    # load config
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')['fnta']
    model_config = load_config('model.yml')['decoder_only_model']
    train_config = load_config('train.yml')['decoder_only_train']

    # initialize WandB
    wandb.init(
        project='Attention Conditioned VAE (decoder only model)', 
        config={"train_config": train_config, "model_config": model_config}
        )

    # set current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # directories
    os.makedirs(os.path.join(file_config['data']['model'], 'decoder_only_model'), exist_ok=True)
    os.makedirs(os.path.join(file_config['data']['model'], 'decoder_only_model', f"{current_time}"), exist_ok=True)
    model_save_dir = os.path.join(file_config['data']['model'], 'decoder_only_model', f"{current_time}")

    # data files
    train_file = os.path.join(file_config['data']['train'], 'train_FNTA.csv')
    val_file = os.path.join(file_config['data']['val'], 'val_FNTA.csv')

    # train parameters
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    lr = float(train_config['lr'])

    # dataloaders
    train_dataloader = get_dataloader(
        csv_file=train_file,
        smiles_max_len=data_config['smiles_max_len'],
        protein_max_len=data_config['protein_max_len'],
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = get_dataloader(
        csv_file=val_file,
        smiles_max_len=data_config['smiles_max_len'],
        protein_max_len=data_config['protein_max_len'],
        batch_size=batch_size,
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
    docking_model_file = os.path.join(file_config['data']['docking'], model_config['docking_score_regression_file'])
    docking_model_config = load_config('model.yml')['docking_score_regression_model']

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
        af2_max_len=data_config['protein_max_len'],
    ).to(torch.device('cuda'))

    # training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, current_time, torch.device('cuda'))

    trainer.train(epochs, train_dataloader, val_dataloader, model_save_dir)
    