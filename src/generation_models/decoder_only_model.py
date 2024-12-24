import torch
import torch.nn as nn
from .moses_vae import SmilesVAE
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor

"""
DecoderOnlyCVAE model.
moses vae の decoder を docking score predictor の attention map で条件付けして追加学習するためのモデル
注意：docking score predictor は return attn wts が True である必要がある
"""

class DecoderOnlyCVAE(nn.Module):
    def __init__(self, smiles_vae: SmilesVAE, docking_score_predictor: DockingScorePredictor, condition_latent_dim=128, transformer_layer_used=0, smiles_max_len=100, af2_max_len=1390):
        super(DecoderOnlyCVAE, self).__init__()
        self.transformer_layer_used = transformer_layer_used
        self.vae_latent_dim = smiles_vae.config['latent_size']
        self.condition_latent_dim = condition_latent_dim

        self.pretrained_smiles_vae = smiles_vae
        self.pretrained_docking_score_predictor = docking_score_predictor

        # freeze the pretrained models
        for param in self.pretrained_smiles_vae.parameters():
            param.requires_grad = False
        for param in self.pretrained_docking_score_predictor.parameters():
            param.requires_grad = False
        for param in self.pretrained_smiles_vae.decoder.parameters():
            param.requires_grad = True

        # 3層の mlp で flatten された attention map を圧縮する
        # (batch_size, smiles_max_len * af2_max_len) -> (batch_size, condition_latent_dim)
        self.mlp_attention_compressor = nn.Sequential(
            nn.Linear(smiles_max_len * af2_max_len, 10000), # 139000 -> 10000
            nn.ReLU(),
            nn.Linear(10000, 1024), # 10000 -> 1024
            nn.ReLU(),
            nn.Linear(1024, self.condition_latent_dim), # 1024 -> condition_latent_dim
        )

        # z-dnn: 連結された潜在変数を vae のデコーダーの入力に変換する
        # (batch_size, vae_latent_dim + condition_latent_dim) -> (batch_size, vae_latent_dim)
        self.concat_latent_dim = self.vae_latent_dim + self.condition_latent_dim
        decrease_dim = self.concat_latent_dim // 3
        self.z_dnn = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim - decrease_dim),
            nn.Linear(self.concat_latent_dim - decrease_dim, 
                      self.concat_latent_dim - decrease_dim * 2),
            nn.Linear(self.concat_latent_dim - decrease_dim * 2,
                      self.vae_latent_dim),
        )

    def forward(self, smiles_tensor, chemberta_smiles_embedding, af2_embedding):
        """
        input:
            - smiles_tensor: list(len= batch_size, entry=tensor(len=SMILESの長さ+2, dtype=torch.int))
                - moses vae の forward_encoder に入力する smiles tensor
                    - smiles の one-hot encoding 
                    - embedding に変換するのは moses vae の forward_encoder で行う
            - chemberta_smiles_embedding: tensor.shape(batch_size, smiles_max_len, embed_dim)
                - prediction model に入力する SMILES embedding
            - af2_embedding: tensor.tensor.shape(batch_size, af2_max_len, embed_dim)
                - prediction model に入力する AlphaFold2 embedding
        """

        # SMILES を潜在変数にエンコード
        z_vae, _ = self.pretrained_smiles_vae.forward_encoder(smiles_tensor)

        # docking score predictor で attention map を取得
        _, _, cross_attn_wts = self.pretrained_docking_score_predictor(
            chemberta_smiles_embedding, 
            af2_embedding
            )[self.transformer_layer_used]
        
        # attention map を圧縮して condition を作成
        # (batch_size, smiles_max_len * af2_max_len) -> (batch_size, condition_latent_dim)
        z_condition = self.mlp_attention_compressor(cross_attn_wts.flatten(start_dim=1))

        # z_vae と z_condition を連結
        # z_concat: tensor(batch_size, vae_latent_dim + condition_latent_dim)
        z_concat = torch.cat([z_vae, z_condition], dim=1)

        # z_dnn で z_concat を vae デコーダーの入力に変換
        # z: tensor(batch_size, vae_latent_dim)
        z = self.z_dnn(z_concat)

        # デコードしてロスを計算
        recon_loss = self.pretrained_smiles_vae.forward_decoder(smiles_tensor, z)

        return recon_loss
    
    def generate(self, batch_size, smiles_embedding, af2_embedding):
        """
        ノイズとドッキングスコアのよい af2 embedding と smiles embedding のペアから SMILES を生成する
        """
        with torch.no_grad():
            z_noise = self.pretrained_smiles_vae.sample_z_prior(batch_size=batch_size)

            _, _, cross_attn_wts = self.pretrained_docking_score_predictor(
                smiles_embedding, 
                af2_embedding
                )[self.transformer_layer_used]
            z_condition = self.mlp_attention_compressor(cross_attn_wts.flatten(start_dim=1))

            z_concat = torch.cat([z_noise, z_condition], dim=1)
            z = self.z_dnn(z_concat)

            generated_smiles = self.pretrained_smiles_vae.sample_smiles_from_z(z)

        return generated_smiles
            

            