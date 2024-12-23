import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.num_heads = num_heads
        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=False)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        # cross attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=False)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)

        # feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, smiles_embedding, af2_embedding):
        """
        Forward pass for the TransformerBlock.
        """
        # self attention
        self_attn_output, self_attn_wts = self.self_attn(query=smiles_embedding,
                                             key=smiles_embedding,
                                             value=smiles_embedding, 
                                             )
        self_attn_output = self.self_attn_layer_norm(self_attn_output + smiles_embedding)

        # cross attention
        cross_attn_output, cross_attn_wts = self.cross_attn(query=self_attn_output, 
                                               key=af2_embedding, 
                                               value=af2_embedding, 
                                               )
        cross_attn_output = self.cross_attn_layer_norm(cross_attn_output + self_attn_output)

        # feedforward
        ffn_output = self.ffn(cross_attn_output)
        output = self.ffn_layer_norm(ffn_output + cross_attn_output)

        output = torch.nan_to_num(output, nan=0.0)

        return output, self_attn_wts, cross_attn_wts
    