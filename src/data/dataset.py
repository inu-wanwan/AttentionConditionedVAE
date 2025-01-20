import os
import pandas as pd
import torch
import yaml
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import logging 
from transformers import logging as transformers_logging
"""
dataset.py
Defines the custom dataset class for the SMILES dataset.
"""

# Set logging level to ERROR to avoid unnecessary outputs

transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class SmilesProteinDataset(Dataset):
    def __init__(self, csv_file, smiles_max_len, protein_max_len, 
                 alphafold_dir="data/alphafold", 
                 smiles_column="Canonical_SMILES", 
                 protein_column="Target", 
                 score_column="Docking_score",
                 ligand_id=None):
        """
        Args:
            csv_file (string): Path to the CSV file with SMILES strings and docking scores.
            alphafold_dir (string): Path to the directory containing the AlphaFold embedding.
            smiles_column (string): Name of the column containing the SMILES strings.
            protein_column (string): Name of the column containing the protein names.
            score_column (string): Name of the column containing the docking scores.
        """
        self.smiles_df = pd.read_csv(csv_file)
        self.alphafold_dir = alphafold_dir
        self.smiles_column = smiles_column
        self.protein_column = protein_column
        self.score_column = score_column
        self.smiles_max_len = smiles_max_len
        self.protein_max_len = protein_max_len

        if ligand_id:
            self.smiles_df = self.smiles_df[self.smiles_df["Ligand_id"] == ligand_id]

    def __len__(self):
        return len(self.smiles_df)
    
    def __getitem__(self, idx):
        """
        Retrieve an item by idx
        """
        row = self.smiles_df.iloc[idx]
        smiles = row[self.smiles_column]
        docking_score = row[self.score_column]
        protein_id = row[self.protein_column]

        uniprot_id = row["UniProt ID"]

        protein_embedding, protein_mask = self.load_alphafold_embedding(uniprot_id)

        docking_score = torch.tensor(docking_score, dtype=torch.float32)

        return {
            "smiles": smiles,
            "protein_embedding": protein_embedding,
            "protein_mask": protein_mask,
            "docking_score": docking_score,
            "protein_id": protein_id,
        }
    
    def load_alphafold_embedding(self, uniprot_id):
        """
        AlphaFoldの埋め込みを読み込む関数
        
        Args:
            - protein_name (str): タンパク質の名前
        Returns:
            - torch.tensor: タンパク質の埋め込み
                Shape: (1, max_seq_len, embedding_dim)
        """
        embedding_path = os.path.join(self.alphafold_dir, uniprot_id, "structure.npy")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        embedding = np.load(embedding_path)

        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        padded_embedding = torch.zeros(self.protein_max_len, embedding_tensor.size(-1))
        padded_embedding[:embedding_tensor.size(0), :] = embedding_tensor

        # generate mask
        attention_mask = torch.zeros(self.protein_max_len, dtype=torch.float32)
        attention_mask[:embedding_tensor.size(0)] = 1

        return padded_embedding, attention_mask
    
    def get_ligand_atoms(smiles_list):
        """
        Get the atom embeddings for SMILES strings.
        """
        
    
    @staticmethod
    def featurize_smiles_static(smiles_list, smiles_max_len):
        """
        Static method for batch featurization of SMILES strings.
        This allows featurization without an instance.
        """
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Tokenization
        encoded_inputs = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        # Model inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Pad embeddings to max_seq_len
        token_embeddings = outputs.last_hidden_state
        max_seq_len = smiles_max_len
        embedding_dim = token_embeddings.size(-1)
        padded_embeddings = torch.zeros(len(smiles_list), max_seq_len, embedding_dim, device=device)
        padded_embeddings[:, :token_embeddings.size(1), :] = token_embeddings
        padded_attention_mask = torch.zeros(len(smiles_list), max_seq_len, dtype=attention_mask.dtype, device=device)
        padded_attention_mask[:, :attention_mask.size(1)] = attention_mask

        return padded_embeddings, padded_attention_mask
    
    @staticmethod
    def get_smiles_cls_embeddings(smiles_list):
        """
        Get the class token for SMILES strings.
        """
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Tokenization
        encoded_inputs = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")

        # Model inference
        outputs = model(**encoded_inputs)

        # Extract the class token
        cls_embeddings = outputs.pooler_output

        return cls_embeddings

def load_config(config_file):
    config_path = os.path.join('config', config_file)
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config

