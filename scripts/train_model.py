import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils import load_config
from torch.utils.data import DataLoader
from data.dataset import get_dataloader
from utils import load_config

# load the configuration
config = load_config('filepath.yml')
alphafold_dir = config['data']['alphafold']

dataset = SmilesProteinDataset(
    csv_file='data/samples/sample_dataset.csv',
    alphafold_dir=alphafold_dir,
    )

print(f"Dataset size: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx}")
    print(f"    SMILES: {batch['smiles_embedding'].shape}")
    print(f"    SMILES mask: {batch['smiles_mask'].shape}")
    print(f"    Protein embedding: {batch['protein_embedding'].shape}")
    print(f"    Docking score: {batch['docking_score']}")
