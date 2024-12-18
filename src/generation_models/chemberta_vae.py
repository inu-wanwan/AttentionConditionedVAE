import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ChemBERTaEncoder(nn.Module):
    def __init__(self, embed_dim, latent_dim):
        super(ChemBERTaEncoder, self).__init__()

        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_log_var = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var
    
class GRUDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, dropout):
        super(GRUDecoder, self).__init__()
        
        self.gru = 