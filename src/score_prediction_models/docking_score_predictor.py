import torch 
import torch.nn as nn
from .transformer_block import TransformerBlock

class DockingScorePredictor(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, num_transformer_blocks, smiles_max_len, regressor_hidden_dim=512, dropout=0.1):
        super(DockingScorePredictor, self).__init__()

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * smiles_max_len, regressor_hidden_dim),
            nn.ReLU(),
            nn.Linear(regressor_hidden_dim, 1),
        )

    def forward(self, smiles_embedding, af2_embedding, smiles_mask, af2_mask):
        """
        Forward pass for the DockingScorePredictor.
        """
        # stack transformer blocks
        for block in self.transformer_blocks:
            smiles_embedding = block(smiles_embedding, af2_embedding, smiles_mask, af2_mask)
        
        # flatten the embeddings
        flattened_embedding = smiles_embedding.flatten(start_dim=1) # (batch_size, embed_dim * smiles_max_len)

        docking_score = self.regressor(flattened_embedding)

        return docking_score