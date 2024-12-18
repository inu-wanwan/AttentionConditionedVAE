import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemBERTaEncoder(nn.Module):
    def __init__(self,embed_dim, latent_dim)
        super(ChemBERTaEncoder, self).__init__()

        self.fc_mu = nn.