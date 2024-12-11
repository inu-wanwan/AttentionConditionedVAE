from torch.utils.data import DataLoader
from src.data.dataset import SmilesProteinDataset
import torch

"""
dataloader.py
Defines the custom DataLoader for the SMILES dataset.
ATTENTION: Define the smiles_max_len in this script. please edit if necessary.
"""

    
def custom_colleate_fn(batch):
    """
    Custom collate function for the DataLoader.
    """
    smiles_list = [item["smiles"] for item in batch]
    docking_scores = torch.stack([item["docking_score"] for item in batch])
    protein_embeddings = torch.stack([item["protein_embedding"] for item in batch])
    protein_masks = torch.stack([item["protein_mask"] for item in batch])

    # Batch SMILES featurization
    smiles_embeddings, smiles_mask = SmilesProteinDataset.featurize_smiles_static(smiles_list, smiles_max_len=100)

    return {
        "smiles_embedding": smiles_embeddings,
        "smiles_mask": smiles_mask,
        "protein_embedding": protein_embeddings,
        "protein_mask": protein_masks,
        "docking_score": docking_scores,
    }

def get_dataloader(csv_file, smiles_max_len, protein_max_len, batch_size=32, 
                   shuffle=True, num_workers=4):
    """
    Initialize DataLoader with the dataset and custom collate function.

    Args:
        csv_file (str): Path to the CSV file with SMILES strings and docking scores.
        max_seq_len (int): Maximum sequence length for padding.
        smiles_column (str): Name of the column containing SMILES strings.
        batch_size (int): Number of samples per batch. Default is 16.
        shuffle (bool): Whether to shuffle the data. Default is True.
        num_workers (int): Number of CPU workers for data loading. Default is 4.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = SmilesProteinDataset(
        csv_file=csv_file,
        smiles_max_len=smiles_max_len,
        protein_max_len=protein_max_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_colleate_fn,
    )

    return dataloader