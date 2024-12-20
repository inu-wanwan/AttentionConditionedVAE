import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import wandb
import os
import pandas as pd
from src.generation_models.moses_vae import SmilesVAE
from utils import load_config
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from collections import UserList, defaultdict
from sklearn.model_selection import train_test_split

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


class KLAnnealer:
    def __init__(self, n_epoch, config):
        self.i_start = config["kl_start"]
        self.w_start = config["kl_w_start"]
        self.w_max = config["kl_w_end"]
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer, config):
        self.n_period = config["lr_n_period"]
        self.n_mult = config["lr_n_mult"]
        self.lr_end = config["lr_end"]

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [
            self.lr_end
            + (base_lr - self.lr_end)
            * (1 + math.cos(math.pi * self.current_epoch / self.t_end))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end


class Trainer:
    def __init__(self, config) -> None:
        self.config = config

    def _train_epoch(self, model, epoch, tqdm_data, kl_weight, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        kl_loss_values = CircularBuffer(self.config["n_last"])
        recon_loss_values = CircularBuffer(self.config["n_last"])
        loss_values = CircularBuffer(self.config["n_last"])
        for batch_idx, input_batch in enumerate(tqdm_data):
            input_batch = tuple(data.to(model.device) for data in input_batch)

            # Forward
            kl_loss, recon_loss = model(input_batch)
            loss = kl_weight * kl_loss + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(model), self.config["clip_grad"])
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [
                f"loss={loss_value:.5f}",
                f"(kl={kl_loss_value:.5f}",
                f"recon={recon_loss_value:.5f})",
                f"klw={kl_weight:.5f} lr={lr:.5f}",
            ]
            tqdm_data.set_postfix_str(" ".join(postfix))

            # Log to WandB
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                "kl_weight": kl_weight,
                "lr": lr,
                "kl_loss": kl_loss.item(),
                "recon_loss": recon_loss.item(),
                "loss": loss.item(),
                "mode": "Eval" if optimizer is None else "Train",
            })

        postfix = {
            "epoch": epoch,
            "kl_weight": kl_weight,
            "lr": lr,
            "kl_loss": kl_loss_value,
            "recon_loss": recon_loss_value,
            "loss": loss_value,
            "mode": "Eval" if optimizer is None else "Train",
        }

        return postfix

    def get_optim_params(self, model):
        return (p for p in model.vae.parameters() if p.requires_grad)

    def _train(self, model, train_loader, val_loader=None, logger=None):
        device = model.device
        n_epoch = self._n_epoch()

        optimizer = optim.Adam(self.get_optim_params(model), lr=self.config["lr_start"])
        kl_annealer = KLAnnealer(n_epoch, self.config)
        lr_annealer = CosineAnnealingLRWithRestart(optimizer, self.config)

        model.zero_grad()
        for epoch in range(n_epoch):
            # Epoch start
            kl_weight = kl_annealer(epoch)
            tqdm_data = tqdm(train_loader, desc="Training (epoch #{})".format(epoch))
            postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config["log_file"])

            if val_loader is not None:
                tqdm_data = tqdm(
                    val_loader, desc="Validation (epoch #{})".format(epoch)
                )
                postfix = self._train_epoch(model, epoch, tqdm_data, kl_weight)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config["log_file"])

            if (self.config["model_save"] is not None) and (
                epoch % self.config["save_frequency"] == 0
            ):
                model = model.to("cpu")
                torch.save(
                    model.state_dict(),
                    self.config["model_save"][:-3] + "_{0:03d}.pt".format(epoch),
                )
                model = model.to(device)

            # Epoch end
            wandb.log(postfix)
            lr_annealer.step()

    def fit(self, model, train_dataloader, val_loader=None):
        # logger = Logger() if self.config["log_file is not None else None
        logger = None

        # train_loader = self.get_dataloader(model, train_data, shuffle=True)
        train_loader = train_dataloader
        """
        val_loader = (
            None
            if val_data is None
            else self.get_dataloader(model, val_data, shuffle=False)
        )
        """

        self._train(model, train_loader, val_loader, logger)
        return model

    def _n_epoch(self):
        return sum(
            self.config["lr_n_period"] * (self.config["lr_n_mult"] ** i)
            for i in range(self.config["lr_n_restarts"])
        )


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



if __name__ == "__main__":
    # load config
    file_config = load_config("filepath.yml")

    # data files
    row_file = os.path.join(file_config["data"]["preprocessed"], "filtered_chembl_35.csv")
    train_file = os.path.join(file_config["data"]["train"], "chembl_35_train.csv")
    val_file = os.path.join(file_config["data"]["val"], "chembl_35_val.csv")

    
    # split data
    df = pd.read_csv(row_file)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df.to_csv(os.path.join(file_config["data"]["train"], "chembl_35_train.csv"), index=False)
    print(f"Train data is saved at {train_file}")
    val_df.to_csv(os.path.join(file_config["data"]["val"], "chembl_35_val.csv"), index=False)
    print(f"Validation data is saved at {val_file}")
    


    # output file 
    save_file = os.path.join(file_config["model"], "smiles_vae_chembl_train_smiles_no_dot.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_smi_path = "../data/drd2_train_smiles_no_dot.smi"
    save_model_path = "smiles_vae_drd2_train_smiles_no_dot.pt"

    # train_smi_path = "../data/Druglike_million_canonical_no_dot_dup.smi"
    # save_model_path = "smiles_vae_dmqp1m_no_dot_dup.pt"

    config = {
        "encoder_hidden_size": 256,  # encoderのGRUの隠れ層の次元数h
        "encoder_num_layers": 1,  # encoderのGRUの層数
        "bidirectional": True,  # Trueなら双方向，Falseなら単方向
        "encoder_dropout": 0.5,  # encoderのGRUのdropout率
        "latent_size": 128,  # 潜在変数の次元数z
        "decoder_hidden_size": 512,  # decoderのGRUの隠れ層の次元数h
        "decoder_num_layers": 3,  # decoderのGRUの層数
        "decoder_dropout": 0,  # decoderのGRUのdropout率
        "n_batch": 128,  # バッチサイズ
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

    # メモリ切れ対策
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
        
    train_smiles = read_smiles(train_file)
    val_smiles = read_smiles(val_file)

    # initialize WandB
    wandb.init(project="moses_vae_model", config=config)

    print("vocabの作成")
    vocab = make_vocab(train_smiles)

    print("モデルの作成")
    model = SmilesVAE(vocab, config, device).to(device)

    print("トレーニングデータローダーの作成")
    train_smiles_tqdm = tqdm(train_smiles, desc="Tokenizing train smiles")
    train = [model.string2tensor(smiles) for smiles in train_smiles_tqdm]

    print("バリデーションデータローダーの作成")
    val_smiles_tqdm = tqdm(val_smiles, desc="Tokenizing val smiles")
    val = [model.string2tensor(smiles) for smiles in val_smiles_tqdm]

    print("trainのpadding")
    train = torch.nn.utils.rnn.pad_sequence(
        train, batch_first=True, padding_value=model.PAD
    )

    print("valのpadding")
    val = torch.nn.utils.rnn.pad_sequence(
        val, batch_first=True, padding_value=model.PAD
    )

    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=config["n_batch"], shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val, batch_size=config["n_batch"], shuffle=True
    )

    print("学習開始")
    trainer = Trainer(config)
    model = trainer.fit(model, train_dataloader, val_loader=val_dataloader)

    # モデルのサマリーを出力
    print(model)

    # モデルの保存
    torch.save(model.state_dict(), save_file)

    print(f"学習データとして{train_smi_path}を利用したモデルを{save_file}として保存しました．")
