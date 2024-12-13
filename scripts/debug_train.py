import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils import load_config
from torch.utils.data import DataLoader
from src.data.dataloader import get_dataloader
from utils import load_config
from tqdm import tqdm

# load config
file_config = load_config('filepath.yml')
data_config = load_config('data.yml')
model_config = load_config('model.yml')['sample_model']
train_config = load_config('train.yml')['sample_model']

# directories
alphafold_dir = file_config['data']['alphafold']
data_dir = file_config['data']['samples']

# data files
train_file = os.path.join(file_config['data']['train'], 'train.csv')
val_file = os.path.join(file_config['data']['val'], 'val.csv')
test_file = os.path.join(file_config['data']['test'], 'test.csv')

# train parameters
batch_size = train_config['batch_size']
epochs = train_config['epochs']
lr = float(train_config['lr'])

# dataloaders
train_dataloader = get_dataloader(
    csv_file=train_file,
    smiles_max_len=data_config['dataset']['smiles_max_len'],
    protein_max_len=data_config['dataset']['protein_max_len'],
    batch_size=batch_size,
    shuffle=True,
)

val_dataloader = get_dataloader(
    csv_file=val_file,
    smiles_max_len=data_config['dataset']['smiles_max_len'],
    protein_max_len=data_config['dataset']['protein_max_len'],
    batch_size=batch_size,
    shuffle=False,
)

train_pbar = tqdm(train_dataloader, desc="Training")
for batch_idx, batch in enumerate(train_pbar):
    print(f"Batch {batch_idx}")
    print(f"    SMILES: {batch['smiles_embedding'].shape}")
    print(f"    SMILES mask: {batch['smiles_mask'].shape}")
    print(f"    Protein embedding: {batch['protein_embedding'].shape}")
    print(f"    Docking score: {batch['docking_score']}")
    break

