import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from .transformer_block import TransformerBlock
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from schnetpack.representation import SchNet
from schnetpack.nn.radial import GaussianRBF
from schnetpack.nn.cutoff import CosineCutoff

class SchNetDockingScorePredictor(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            ffn_hidden_dim, 
            num_transformer_blocks, 
            schnet_n_interactions=3,
            schnet_cutoff=5.0,
            atoms_max_len=350,
            regressor_hidden_dim=512,
            dropout=0.1,
            return_attn_wts=False
        ):
        super(SchNetDockingScorePredictor, self).__init__()

        self.return_attn_wts = return_attn_wts

        self.embed_dim = embed_dim

        self.atoms_max_len = atoms_max_len

        cutoff = schnet_cutoff
        radial_basis = GaussianRBF(n_rbf=20, cutoff=cutoff)

        cutoff_fn = CosineCutoff(cutoff)

        self.schnet = SchNet(
            n_atom_basis=embed_dim,
            n_interactions=schnet_n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
        )

        self.af2_embedding_dim_reducer = None
        if embed_dim != 385:
            self.af2_embedding_dim_reducer = nn.Sequential(
                nn.Linear(385, embed_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * atoms_max_len, regressor_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(regressor_hidden_dim, 1),
        )
    
    def pad_schnet_embeddings(self, schnet_embeddings, atoms_max_len):
        """
        Pad the SchNet embeddings to the maximum number of atoms.
        """
        padded_schnet_embeddings = []
        for emb in schnet_embeddings:
            num_atoms, emb_dim = emb.shape
            
            padded_emb = F.pad(
                emb,
                (0, 0, 0, atoms_max_len - num_atoms),
                mode='constant',
                value=0
            )
            padded_schnet_embeddings.append(padded_emb)

        return torch.stack(padded_schnet_embeddings)

    
    def forward(self, batch_positions, batch_atomic_numbers, af2_embedding):
        """
        Forward pass for the SchNetDockingScorePredictor.
        Arguments:
            batch_positions (torch.Tensor): Tensor of atomic positions (batch_size, num_atoms, 3).
            batch_atomic_numbers (torch.Tensor): Tensor of atomic numbers (batch_size, num_atoms).
            af2_embedding (torch.Tensor): Tensor of AF2 embeddings (batch_size, embed_dim).
        Returns:
            docking_score (torch.Tensor): Predicted docking scores (batch_size, 1).
        """
        # calculate SchNet embeddings
        schnet_embeddings = self.schnet(
            positions=batch_positions,
            atomic_numbers=batch_atomic_numbers
        )

        # pad the embeddings
        schnet_embeddings = self.pad_schnet_embeddings(schnet_embeddings, self.atoms_max_len) # (batch_size, atoms_max_len, embed_dim)

        # reduce the embedding dimension if necessary
        if self.af2_embedding_dim_reducer:
            af2_embedding = self.af2_embedding_dim_reducer(af2_embedding)
        
        # stack transformer blocks
        if self.return_attn_wts:
            self_attn_wts_list = []
            cross_attn_wts_list = []
            for block in self.transformer_blocks:
                schnet_embeddings, self_attn_wts, cross_attn_wts = block(schnet_embeddings, af2_embedding)
                self_attn_wts_list.append(self_attn_wts)
                cross_attn_wts_list.append(cross_attn_wts)
        else:
            for block in self.transformer_blocks:
                schnet_embeddings, _, _ = block(schnet_embeddings, af2_embedding)

        # flatten the embeddings
        flattened_embedding = schnet_embeddings.flatten(start_dim=1)

        # pass through the regressor
        docking_score = self.regressor(flattened_embedding)

        if self.return_attn_wts:
            return docking_score, self_attn_wts_list, cross_attn_wts_list
        
        return docking_score