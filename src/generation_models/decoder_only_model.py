import torch
import torch.nn as nn
from .moses_vae import SmilesVAE
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor

"""
DecoderOnlyCVAE model.
moses vae の decoder を docking score predictor の attention map で条件付けして追加学習するためのモデル

"""

class DecoderOnlyCVAE(nn.Module):
    def __init__(self, smiles_vae: SmilesVAE, docking_score_predictor: DockingScorePredictor):
        super(DecoderOnlyCVAE, self).__init__()
        self.smiles_vae = smiles_vae
        self.docking_score_predictor = docking_score_predictor

    def forward(self, smiles_embedding, af2_embedding):
        """
        Forward pass for the DecoderOnlyCVAE.
        """
        smiles_embedding = self.smiles_vae.decoder(smiles_embedding)
        docking_score_pred, self_attn_wts_list, cross_attn_wts_list = self.docking_score_predictor(smiles_embedding, af2_embedding)

        return docking_score_pred, self_attn_wts_list, cross_attn_wts_list