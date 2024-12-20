import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from tqdm.auto import tqdm
from typing import List
import math
import numpy as np
import pandas as pd
from collections import UserList, defaultdict
from torch.optim.lr_scheduler import _LRScheduler


class SmilesVAE(nn.Module):
    def __init__(self, vocab, config, device):
        super().__init__()

        self.device = device

        self.char2idx = vocab  # dict of tokens->int. Contains "PAD, SOS, EOS, UNK"
        self.idx2char = {
            i: c for c, i in vocab.items()
        }  # dict of int->tokens. Contains "PAD, SOS, EOS, UNK"
        self.config = config

        # 4つの特殊トークンをこのクラスの属性として追加
        for ss in ["PAD", "SOS", "EOS", "UNK"]:
            setattr(self, ss, self.char2idx[ss])

        embedding_dim = len(self.char2idx) - 1  # vocabの種類数（PADトークンを除く）
        self.embedding = nn.Embedding(
            len(self.char2idx), embedding_dim, padding_idx=self.PAD
        )
        # vocabの種類数，embedding後の次元数，無視するindex

        # encoder
        self.encoder_gru = nn.GRU(
            embedding_dim,
            config["encoder_hidden_size"],
            num_layers=config["encoder_num_layers"],
            batch_first=True,
            bidirectional=config["bidirectional"],
            dropout=config["encoder_dropout"]
            if config["encoder_num_layers"] > 1
            else 0,
        )

        encoder_gru_output_dim = config["encoder_hidden_size"] * (
            2 if config["bidirectional"] else 1
        )

        self.encoder_mean = nn.Linear(encoder_gru_output_dim, config["latent_size"])
        self.encoder_logvar = nn.Linear(encoder_gru_output_dim, config["latent_size"])

        # decoder
        self.decoder_gru = nn.GRU(
            embedding_dim + config["latent_size"],
            config["decoder_hidden_size"],
            num_layers=config["decoder_num_layers"],
            batch_first=True,
            dropout=config["decoder_dropout"]
            if config["decoder_num_layers"] > 1
            else 0,
        )  # input: (batch_size, seq_len, embedding_dim + latent_size),
        # output: (batch_size, seq_len, decoder_hidden_size)

        self.decoder_fc = nn.Linear(
            config["decoder_hidden_size"], len(self.char2idx)
        )  # input: (batch_size, decoder_hidden_size),
        # output: (batch_size, vocab_size)
        # decoderの出力をvocabの種類に変換するための全結合層

        self.decoder_h0 = nn.Linear(
            config["latent_size"], config["decoder_hidden_size"]
        )  # input: (batch_size, latent_size),
        # output: (batch_size, decoder_hidden_size)
        # decoderの初期隠れ層h0を潜在変数zから作成するための全結合層

        self.encoder = nn.ModuleList(
            [self.encoder_gru, self.encoder_mean, self.encoder_logvar]
        )
        self.decoder = nn.ModuleList(
            [self.decoder_gru, self.decoder_fc, self.decoder_h0]
        )
        self.vae = nn.ModuleList([self.embedding, self.encoder, self.decoder])

    def string2tensor(self, string):
        # 文字列をtensorに変換する
        result = [self.SOS]
        for c in string:
            if c in self.char2idx:
                result.append(self.char2idx[c])
            else:
                result.append(self.UNK)
        result.append(self.EOS)
        return torch.tensor(result).to(self.device)  # tensor: (stringの長さ+2)

    def tensor2string(self, tensor):
        # tensorを文字列に変換する
        result = ""
        for i in tensor:
            i = i.item()
            if i == self.SOS or i == self.PAD or i == self.UNK:
                continue
            if i == self.EOS:
                break
            result += self.idx2char[i]
        return result

    def forward(self, x):
        """
        x: list(len= batch_size, entry=tensor(dim=[SMILESの長さ+2,] dtype=torch.int))
        """
        z, kl_loss = self.forward_encoder(x)
        recon_loss = self.forward_decoder(x, z)
        return recon_loss, kl_loss

    def forward_encoder(self, x):
        """
        x: list(len= batch_size, entry=tensor(len=SMILESの長さ+2, dtype=torch.int))
        """
        x = [
            self.embedding(i) for i in x
        ]  # list(len= batch_size, entry=tensor(dim=[各SMILESの長さ+2, embedding_dim]))
        x = nn.utils.rnn.pad_sequence(
            x, batch_first=True, padding_value=self.PAD
        )  # tensor(dim=[batch_size, 最大SMILESの長さ+2, embedding_dim])

        _, h = self.encoder_gru(
            x
        )  # h: (num_layers * num_directions, batch_size, encoder_hidden_size)
        h = h[
            -(1 + int(self.encoder_gru.bidirectional)) :
        ]  # h: (a(1 or 2), batch_size, encoder_hidden_size)

        h = h.split(
            1
        )  # h: list(len=a, entry=tensor(dim=[1, batch_size, encoder_hidden_size]))
        h = torch.cat(h, dim=-1)  # h: (1, batch_size, a*encoder_hidden_size)
        h = h.squeeze(0)  # h: (batch_size, a*encoder_hidden_size)

        mean = self.encoder_mean(h)  # mean: (batch_size, latent_size)
        logvar = self.encoder_logvar(h)  # logvar: (batch_size, latent_size)

        eps = torch.randn_like(mean)
        z = mean + (logvar / 2).exp() * eps

        kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(1).mean()

        return z, kl_loss

    def forward_decoder(self, x, z):
        """
        x: list(len= batch_size, entry=tensor(len=SMILESの長さ+2, dtype=torch.int))
        z: tensor(dim=[batch_size, latent_size])
        """

        length = [len(i) for i in x]
        x = nn.utils.rnn.pad_sequence(
            x, batch_first=True, padding_value=self.PAD
        )  # x: (batch_size, SMILESの最大長+2)
        embedding = self.embedding(x).to(
            self.device
        )  # embedding: tensor(batch_size, SMILESの最大長+2, embedding_dim)

        z_0 = z.unsqueeze(1)  # z: (batch_size, 1, latent_size)
        z_0 = z_0.repeat(
            1, embedding.size(1), 1
        )  # z: (batch_size, SMILESの最大長+2, latent_size)

        input = torch.cat(
            [embedding, z_0], dim=-1
        )  # x_input: (batch_size, SMILESの最大長+2, embedding_dim+latent_size)
        input = nn.utils.rnn.pack_padded_sequence(
            input, length, batch_first=True
        )  # x_input: (batch_size, SMILESの最大長+2, embedding_dim+latent_size)

        h_0 = self.decoder_h0(z)  # h_0: (batch_size, decoder_hidden_size)
        h_0 = h_0.unsqueeze(0)  # h_0: (1, batch_size, decoder_hidden_size)
        h_0 = h_0.repeat(
            self.decoder_gru.num_layers, 1, 1
        )  # h_0: (decoder_num_layers, batch_size, decoder_hidden_size)

        output, _ = self.decoder_gru(
            input, h_0
        )  # output: (batch_size, SMILESの最大長+2, decoder_hidden_size)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.decoder_fc(
            output
        )  # output: (batch_size, SMILESの最大長+2, vocab_size)

        recon_loss = F.cross_entropy(
            output[:, :-1].contiguous().view(-1, output.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.PAD,
        )

        return recon_loss
    
    def forward_decoder_eachloss(self, x, z):
        """
        生成したSMILESのCossEntropyLossの各値とSMILESを返す
        x: list(len= batch_size, entry=tensor(len=SMILESの長さ+2, dtype=torch.int))
        z: tensor(dim=[batch_size, latent_size])
        return: each_recon_loss
        """

        length = [len(i) for i in x]
        x = nn.utils.rnn.pad_sequence(
            x, batch_first=True, padding_value=self.PAD
        )  # x: (batch_size, SMILESの最大長+2)
        embedding = self.embedding(x).to(
            self.device
        )  # embedding: tensor(batch_size, SMILESの最大長+2, embedding_dim)

        z_0 = z.unsqueeze(1)  # z: (batch_size, 1, latent_size)
        z_0 = z_0.repeat(
            1, embedding.size(1), 1
        )  # z: (batch_size, SMILESの最大長+2, latent_size)

        input = torch.cat(
            [embedding, z_0], dim=-1
        )  # x_input: (batch_size, SMILESの最大長+2, embedding_dim+latent_size)
        input = nn.utils.rnn.pack_padded_sequence(
            input, length, batch_first=True
        )  # x_input: (batch_size, SMILESの最大長+2, embedding_dim+latent_size)

        h_0 = self.decoder_h0(z)  # h_0: (batch_size, decoder_hidden_size)
        h_0 = h_0.unsqueeze(0)  # h_0: (1, batch_size, decoder_hidden_size)
        h_0 = h_0.repeat(
            self.decoder_gru.num_layers, 1, 1
        )  # h_0: (decoder_num_layers, batch_size, decoder_hidden_size)

        output, _ = self.decoder_gru(
            input, h_0
        )  # output: (batch_size, SMILESの最大長+2, decoder_hidden_size)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.decoder_fc(
            output
        )  # output: (batch_size, SMILESの最大長+2, vocab_size)
        
        each_recon_loss = []
        for i in range(output.size(0)):
            each_recon_loss.append(
                F.cross_entropy(
                    output[i][:-1],
                    x[i][1:],
                    ignore_index=self.PAD
                    )
                )

        return each_recon_loss
    
    

    def sample_z_prior(self, batch_size):
        return torch.randn(batch_size, self.config["latent_size"])

    def sample_smiles(self, batch_size, max_length=100):
        with torch.no_grad():
            z = self.sample_z_prior(batch_size)
            z = z.to(self.device)

            z_0 = z.unsqueeze(1)  # z: (batch_size, 1, latent_size)

            h = self.decoder_h0(z)
            h = h.unsqueeze(0)
            h = h.repeat(self.decoder_gru.num_layers, 1, 1)

            w = torch.tensor(self.SOS).repeat(batch_size).to(self.device)

            x = torch.tensor(self.PAD).repeat(batch_size, max_length).to(self.device)
            x[:, 0] = self.SOS

            end_pads = (
                torch.tensor([max_length]).repeat(batch_size).to(self.device)
            )  # end_pads: (batch_size)
            eos_mask = torch.zeros(batch_size, dtype=torch.uint8).to(
                self.device
            )  # eos_mask: (batch_size)

            # Generating cycle
            for i in range(1, max_length):
                # batch_size個並べられたsosをEmbeddingしたあと，次元数を増やす
                embedding = self.embedding(w).unsqueeze(
                    1
                )  # embedding: (batch_size, 1, embedding_dim)
                x_input = torch.cat(
                    [embedding, z_0], dim=-1
                )  # x_input: (batch_size, 1, embedding_dim+latent_size)

                o, h = self.decoder_gru(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y, dim=-1)  # y: (batch_size, vocab_size)

                w = torch.multinomial(y, 1)[
                    :, 0
                ]  # w: (batch_size). yの各行に対して，カテゴリ分布に従ってサンプリングした結果を返す
                x[~eos_mask, i] = w[
                    ~eos_mask
                ]  # eos_maskがTrueである行のi列目（=各生成中SMILESのi番目の文字）にサンプリング結果を代入
                i_eos_mask = ~eos_mask & (
                    w == self.EOS
                )  # eos_maskがTrueかつwがEOS（=）ならTrue
                
                end_pads[i_eos_mask] = (
                    i + 1
                )  # i_eos_maskがTrueの行のend_padsの値をi+1にする（=各生成中SMILESのEOSの位置を記録）
                eos_mask = eos_mask | i_eos_mask  # eos_maskとi_eos_maskの論理和をとる

            # xをSMILESに変換
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, : end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]

    def sample_smiles_from_z(self, z, max_length=100):
        with torch.no_grad():
            z = z.to(self.device)
            batch_size = z.size(0)

            z_0 = z.unsqueeze(1)  # z: (batch_size, 1, latent_size)

            h = self.decoder_h0(z)
            h = h.unsqueeze(0)
            h = h.repeat(self.decoder_gru.num_layers, 1, 1)

            w = torch.tensor(self.SOS).repeat(batch_size).to(self.device)

            x = torch.tensor(self.PAD).repeat(batch_size, max_length).to(self.device)
            x[:, 0] = self.SOS

            end_pads = (
                torch.tensor([max_length]).repeat(batch_size).to(self.device)
            )  # end_pads: (batch_size)
            eos_mask = torch.zeros(batch_size, dtype=torch.bool).to(
                self.device
            )  # eos_mask: (batch_size)

            # Generating cycle
            for i in range(1, max_length):
                # batch_size個並べられたsosをEmbeddingしたあと，次元数を増やす
                embedding = self.embedding(w).unsqueeze(
                    1
                )  # embedding: (batch_size, 1, embedding_dim)
                x_input = torch.cat(
                    [embedding, z_0], dim=-1
                )  # x_input: (batch_size, 1, embedding_dim+latent_size)

                o, h = self.decoder_gru(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                
                y = F.softmax(y, dim=-1)  # y: (batch_size, vocab_size)
                # print("in sample_smiles_from_z   y.shape:", y.shape)
                w = torch.multinomial(y, 1)[
                    :, 0
                ]  # w: (batch_size). yの各行に対して，カテゴリ分布に従ってサンプリングした結果を返す
                x[~eos_mask, i] = w[
                    ~eos_mask
                ]  # eos_maskがTrueである行のi列目（=各生成中SMILESのi番目の文字）にサンプリング結果を代入
                i_eos_mask = ~eos_mask & (
                    w == self.EOS
                )  # eos_maskがTrueかつwがEOS（=）ならTrue
                end_pads[i_eos_mask] = (
                    i + 1
                )  # i_eos_maskがTrueの行のend_padsの値をi+1にする（=各生成中SMILESのEOSの位置を記録）
                eos_mask = eos_mask | i_eos_mask  # eos_maskとi_eos_maskの論理和をとる

            # xをSMILESに変換
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, : end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]
        
    def sample_next_char(self, cur_char_tensor, smiles_latent, decoder_h):
        """
        cur_char_tensor: tensor(dim=[batch_size])
        smiles_latent: tensor(dim=[batch_size, latent_size])
        decoder_h: tensor(dim=[batch_size, decoder_hidden_size])
        return: y, next_h (y: tensor(dim=[batch_size, vocab_size]), h: tensor(dim=[batch_size, decoder_hidden_size]))
        """
        cur_char_tensor = cur_char_tensor.to(self.device)
        smiles_latent = smiles_latent.to(self.device)

        z = self.embedding(cur_char_tensor).to(self.device)  # (batch_size, embedding_dim)
        batch_size = z.size(0)
        # print("In 'sample_next_char()',  z.shape:", z.shape)
        if decoder_h is None:
            h = self.decoder_h0(smiles_latent)  # h: (batch_size, decoder_hidden_size)
            h = h.unsqueeze(0)  # h: (1, batch_size, decoder_hidden_size)
            h = h.repeat(self.decoder_gru.num_layers, 1, 1)  # h: (decoder_num_layers, batch_size, decoder_hidden_size)

        else: # h is not None
            h = decoder_h.to(self.device)
        # print("In 'sample_next_char()',  h.shape:", h.shape)

        w = torch.tensor(self.SOS).repeat(batch_size).to(self.device)
        
        # Generating cycle
        z = z.unsqueeze(1)  # embedding: (batch_size, 1, embedding_dim)
        smiles_latent = smiles_latent.unsqueeze(1) # (batch_size, 1, latent_size)
        x_input = torch.cat(
            [z, smiles_latent], dim=-1
        )  # x_input: (batch_size, 1, embedding_dim+latent_size)

        out, next_h = self.decoder_gru(x_input, h)
        y = self.decoder_fc(out.squeeze(1))

        return y, next_h # y: (batch_size, vocab_size)
