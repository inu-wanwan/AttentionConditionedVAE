import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        # cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)

        # feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, smiles_embedding, af2_embedding, smiles_mask=None, af2_mask=None):
        """
        Forward pass for the TransformerBlock.
        """
        # prepare self attention mask
        self_attn_mask = self._prepare_cross_attention_mask(smiles_mask, af2_mask) 

        # self attention
        self_attn_output, _ = self.self_attn(query=smiles_embedding,
                                             key=smiles_embedding,
                                             value=smiles_embedding, 
                                             attn_mask=self_attn_mask)
        self_attn_output = self.self_attn_layer_norm(self_attn_output + smiles_embedding)

        # prepare cross attention mask
        cross_attn_mask = self._prepare_cross_attention_mask(smiles_mask, af2_mask)

        # cross attention
        cross_attn_output, _ = self.cross_attn(query=self_attn_output, 
                                               key=af2_embedding, 
                                               value=af2_embedding, 
                                               attn_mask=cross_attn_mask)
        cross_attn_output = self.cross_attn_layer_norm(cross_attn_output + self_attn_output)

        # feedforward
        ffn_output = self.ffn(cross_attn_output)
        output = self.ffn_layer_norm(ffn_output + cross_attn_output)

        return output
    
    def _prepare_self_attention_mask(self, smiles_mask):
        """
        Prepare the attention mask for the self-attention.
        input mask -> (batch_size, seq_len), 1 or 0
        output -> (batch_size, seq_len, seq_len), 0 or -inf
        """
        smiles_mask = smiles_mask.float()

        attn_mask = smiles_mask.unsqueeze(2) * smiles_mask.unsqueeze(1)

        attn_mask = torch.where(
            smiles_mask > 0,
            torch.tensor(0.0, device=smiles_mask.device),
            torch.tensor(float("-inf"), device=smiles_mask.device),
        )

        return attn_mask
    
    def _prepare_cross_attention_mask(self, smiles_mask, af2_mask):
        """
        Prepare the attention mask for the cross-attention.
        """
        smiles_mask = smiles_mask.float()
        af2_mask = af2_mask.float()

        combined_mask = smiles_mask.unsqueeze(2) * af2_mask.unsqueeze(1)

        attn_mask = torch.where(
            combined_mask > 0,
            torch.tensor(0.0, device=combined_mask.device),
            torch.tensor(float("-inf"), device=combined_mask.device),
        )

        return attn_mask