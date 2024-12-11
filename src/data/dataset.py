import os
import pandas as pd
import torch
import yaml
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
"""
dataset.py
Defines the custom dataset class for the SMILES dataset.
"""

class SmilesProteinDataset(Dataset):
    def __init__(self, csv_file, alphafold_dir, smiles_colmn="Canonical_SMILES", 
                 protein_column="Protein", score_colmn="Docking_score"):
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
        self.smiles_colmn = smiles_colmn
        self.protein_column = protein_column
        self.score_colmn = score_colmn
        uniprot_mapping_file = os.path.join("data", "DUD-E", "targets_pdb_ids.csv")
        mapping_df = pd.read_csv(uniprot_mapping_file)
        self.target_to_uniprot = mapping_df.set_index("Target")["UniProt ID"].to_dict()

    def __len__(self):
        return len(self.smiles_df)
    
    def featurize_smiles(self, smiles_list):
        """
        SMILES のトークンごとの埋め込みを取得する関数
        
        Args:
            - smiles_list (list): SMILESのリスト
        Returns:
            - torch.tensor: SMILESのトークンごとの埋め込み
                Shape: (batch_size, max_seq_len, embedding_dim)
            - torch.tensor: padding mask
                Shape: (batch_size, max_seq_len)
        """
        # tokenizerの読み込み
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        # modelの読み込み
        model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        # デバイスの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # tokenization
        encoded_inputs = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")

        # デバイスに転送
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # get token embeddings
        token_embeddings = outputs.last_hidden_state

        return token_embeddings, attention_mask
    
    def get_uniprot_id(self, target):
        if target.upper() not in self.target_to_uniprot:
            raise KeyError(f"Protein name not found: {target}")
        return self.target_to_uniprot[target.upper()]
    
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
        return torch.tensor(embedding, dtype=torch.float32)
    
    def __getitem__(self, idx):
        """
        Retrieve an item by idx
        """
        row = self.smiles_df.iloc[idx]
        smiles = row[self.smiles_colmn]
        docking_score = row[self.score_colmn]
        protein_id = row[self.protein_column]

        # get the corresponding uniprot id
        try:
            uniprot_id = self.get_uniprot_id(protein_id)
        except KeyError:
            raise KeyError(f"Protein name not found: {protein_id}")
        
        # get the SMILES embedding
        try:
            smiles_embedding, smiles_mask = self.featurize_smiles([smiles])
        except ValueError:
            raise ValueError(f"Error featurizing SMILES: {smiles}")
        
        # load the AlphaFold embedding
        try:
            protein_embedding = self.load_alphafold_embedding(uniprot_id)
        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding file not found: {uniprot_id}")
        
        # Convert the docking score to a tensor
        docking_score = torch.tensor(docking_score, dtype=torch.float32)

        return {
            "smiles_embedding": smiles_embedding,
            "smiles_mask": smiles_mask,
            "protein_embedding": protein_embedding,
            "docking_score": docking_score,
        }

def load_config(config_file):
    config_path = os.path.join('config', config_file)
    with open(config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    return config

