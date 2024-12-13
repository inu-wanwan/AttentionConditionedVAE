import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils import load_config
from torch.utils.data import DataLoader
from src.data.dataloader import get_dataloader
from utils import load_config

# load the configuration
file_config = load_config('filepath.yml')
data_config = load_config('data.yml')
model_config = load_config('model.yml')

alphafold_dir = file_config['data']['alphafold']
data_dir = file_config['data']['samples']

smiles_max_len = data_config['dataset']['smiles_max_len']
protein_max_len = data_config['dataset']['protein_max_len']

batch_size = model_config['sample_model']['batch_size']

dataloader = get_dataloader(
    csv_file=os.path.join(data_dir, 'sample_dataset.csv'),
    smiles_max_len=smiles_max_len,
    protein_max_len=protein_max_len,
    batch_size=batch_size,
    shuffle=True,
)

for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx}")
    print(f"    SMILES: {batch['smiles_embedding'].shape}")
    print(f"    SMILES mask: {batch['smiles_mask'].shape}")
    print(f"    SMILES mask: {batch['smiles_mask'][0]}")
    print(f"    Protein embedding: {batch['protein_embedding'].shape}")
    print(f"    Protein mask: {batch['protein_mask'].shape}")
    print(f"    Protein mask; {batch['protein_mask'][0]}")
    print(f"    Docking score: {batch['docking_score']}")
    break
