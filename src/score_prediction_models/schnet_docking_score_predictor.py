import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import schnetpack as spk
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
        if embed_dim != 384:
            self.af2_embedding_dim_reducer = nn.Sequential(
                nn.Linear(384, embed_dim, bias=True),
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
    
    def reshape_schnet_embeddings(
            self,
            schnet_embeddings: torch.Tensor,
            idx_m: torch.Tensor,
            batch_size: int,
            atoms_max_len: int
        ) -> torch.Tensor:
        """
        Reshape SchNet embeddings from (atoms_count, embed_dim) to (batch_size, atoms_len, embed_dim).
        
        Args:
            schnet_embeddings (torch.Tensor): Tensor of shape (atoms_count, embed_dim), 
                                            containing atomic embeddings.
            idx_m (torch.Tensor): Tensor of shape (atoms_count,), indicating batch indices for each atom.
            batch_size (int): Total number of batches.
            atoms_len (int): Maximum number of atoms per batch.
            
        Returns:
            torch.Tensor: Reshaped embeddings of shape (batch_size, atoms_len, embed_dim).
        """
        # 埋め込みの次元数を取得
        embed_dim = schnet_embeddings.size(1)
        
        # 出力テンソルを初期化 (パディング部分は 0)
        reshaped_embeddings = torch.zeros(batch_size, atoms_max_len, embed_dim, device=schnet_embeddings.device)
        
        # 各バッチごとに原子の埋め込みを配置
        for i in range(batch_size):
            # 現在のバッチに属する原子をフィルタリング
            mask = idx_m == i
            embeddings_for_batch = schnet_embeddings[mask]  # (N_atoms_in_batch, embed_dim)
            
            # 最大原子数 (atoms_len) に揃えて格納
            reshaped_embeddings[i, :embeddings_for_batch.size(0), :] = embeddings_for_batch
        
        return reshaped_embeddings

    
    def forward(self, batch_schnet_input, af2_embedding):
        """
        Forward pass for the SchNetDockingScorePredictor.
        Arguments:
            batch_schnet_input (dict): Dictionary containing the input tensors for SchNet.
            af2_embedding (torch.Tensor): Tensor of AF2 embeddings (batch_size, embed_dim).
        Returns:
            docking_score (torch.Tensor): Predicted docking scores (batch_size, 1).
        """
        # get batch size
        batch_size = af2_embedding.size(0)

        # calculate SchNet embeddings
        schnet_embeddings = self.schnet(batch_schnet_input)["scalar_representation"] # (batch_size, atoms_max_len, embed_dim)

        # reshape the embeddings
        schnet_embeddings = self.reshape_schnet_embeddings(schnet_embeddings, batch_schnet_input['_idx_m'], batch_size, self.atoms_max_len)

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