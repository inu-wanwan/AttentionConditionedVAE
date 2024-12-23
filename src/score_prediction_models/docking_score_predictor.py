import torch 
import torch.nn as nn
from .transformer_block import TransformerBlock

class DockingScorePredictor(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, num_transformer_blocks, smiles_max_len=100, regressor_hidden_dim=512, dropout=0.1, return_attn_wts=False):
        super(DockingScorePredictor, self).__init__()

        self.return_attn_wts = return_attn_wts

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * smiles_max_len, regressor_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(regressor_hidden_dim, 1),
        )

    def forward(self, smiles_embedding, af2_embedding):
        """
        Forward pass for the DockingScorePredictor.
        """
        # stack transformer blocks
        if self.return_attn_wts:
            self_attn_wts_list = []
            cross_attn_wts_list = []
            for block in self.transformer_blocks:
                smiles_embedding, self_attn_wts, cross_attn_wts = block(smiles_embedding, af2_embedding)
                self_attn_wts_list.append(self_attn_wts)
                cross_attn_wts_list.append(cross_attn_wts)
        else:
            for block in self.transformer_blocks:
                smiles_embedding, _, _ = block(smiles_embedding, af2_embedding)
        
        # flatten the embeddings
        flattened_embedding = smiles_embedding.flatten(start_dim=1) # (batch_size, embed_dim * smiles_max_len)

        docking_score = self.regressor(flattened_embedding)

        if self.return_attn_wts:
            return docking_score, self_attn_wts_list, cross_attn_wts_list

        return docking_score