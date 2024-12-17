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

    def forward(self, smiles_embedding, af2_embedding, smiles_mask, af2_mask):
        """
        Forward pass for the TransformerBlock.
        """
        # prepare key padding mask
        smiles_key_padding_mask = self._prepare_key_padding_mask(smiles_mask)
        af2_key_padding_mask = self._prepare_key_padding_mask(af2_mask)

        # prepare self attention mask
        self_attn_mask = self._prepare_attention_mask(smiles_mask, smiles_mask, self.num_heads) 

        # self attention
        self_attn_output, weight = self.self_attn(query=smiles_embedding,
                                             key=smiles_embedding,
                                             value=smiles_embedding, 
                                             )
        self_attn_output = self.self_attn_layer_norm(self_attn_output + smiles_embedding)

        # prepare cross attention mask
        cross_attn_mask = self._prepare_attention_mask(smiles_mask, af2_mask, num_heads=self.num_heads)

        # cross attention
        cross_attn_output, _ = self.cross_attn(query=self_attn_output, 
                                               key=af2_embedding, 
                                               value=af2_embedding, 
                                               )
        cross_attn_output = self.cross_attn_layer_norm(cross_attn_output + self_attn_output)

        # feedforward
        ffn_output = self.ffn(cross_attn_output)
        output = self.ffn_layer_norm(ffn_output + cross_attn_output)

        output = torch.nan_to_num(output, nan=0.0)

        return output
    
    def _prepare_key_padding_mask(self, mask):
        """
        Prepare key padding mask for MultiheadAttention.
        
        Args:
            mask (torch.Tensor): Mask for embeddings, shape (N, L).
        
        Returns:
            (N, L). value 0, 1 -> True, False
        """
        return ~mask.bool()
    
    def _prepare_attention_mask(self, query_mask, key_mask, num_heads):
        """
        Prepare an attention mask for MultiheadAttention.
        
        Args:
            quert_mask (torch.Tensor): Mask for Query embeddings, shape (N, L_query).
            key_mask (torch.Tensor): Mask for Key embeddings, shape (N, L_key).
            num_heads (int): Number of attention heads.
        
        Returns:
            (N * num_heads, L_query, L_key). 
            value 0, 1 -> True, False
        """
        combined_mask = query_mask.unsqueeze(2) * key_mask.unsqueeze(1)
        
        attn_mask = ~combined_mask.bool()

        batch_size, L_query, L_key = attn_mask.shape
        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, num_heads, L_query, L_key).reshape(batch_size * num_heads, L_query, L_key)
        return attn_mask