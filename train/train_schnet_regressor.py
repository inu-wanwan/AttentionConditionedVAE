import os
import wandb
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import schnetpack as spk
import schnetpack.transform as trn
import schnetpack.properties as properties
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from scripts.utils import load_config
from src.data.dataloader import get_dataloader
from src.score_prediction_models.schnet_docking_score_predictor import SchNetDockingScorePredictor
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', '-c', type=str, default='config/', help='Directory containing config files')
    parser.add_argument('--train_config', '-t', type=str, default='train.yml', help='Training config file')
    parser.add_argument('--model_config', '-m', type=str, default='model.yml', help='Model config file')
    parser.add_argument('--epochs', '-e', type=int, default=None, help='Override the number of epochs in the config file')
    parser.add_argument('--batch_size', '-b', type=int, default=None, help='Override the batch size in the config file')
    parser.add_argument('--lr', '-l', type=float, default=None, help='Override the learning rate in the config file')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID to use')
    return parser.parse_args()

class Trainer:
    def __init__(self, model, optimizer, current_time, device, atoms_max_len):
        self.model = model
        self.optimizer = optimizer
        self.current_time = current_time
        self.device = device
        self.criterion = nn.MSELoss()
        self.atoms_max_len = atoms_max_len

    def smiles_to_atoms(self, smiles):
        """
        Convert SMILES to ASE Atoms object.
        
        Args:
        - smiles (str): SMILES string

        Returns:
        - ase.Atoms or None: ASE Atoms object if successful, None otherwise
        """
        try:
            # convert SMILES to RDKit Mol object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            # generate 3D coordinates
            mol = Chem.AddHs(mol) # add hydrogens
            success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if success != 0:
                raise ValueError(f"Failed to generate 3D coordinates for SMILES: {smiles}")
            
            # energy minimize
            AllChem.UFFOptimizeMolecule(mol)
            
            # get 3d coordinates
            conformer = mol.GetConformer()
            positions = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

            # get atomic numbers
            atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

            # create ASE Atoms object
            atoms = Atoms(
                positions=positions,
                numbers=atomic_numbers
            )

            return atoms
        
        except Exception as e:
            print(f"Error converting SMILES to atoms: {e}")
            return None
    
    def calculate_Rij(self, inputs):
        """
        Calculate Rij 
        Args:
            inputs (dict): Input dictionary containing properties.R (atomic positions) and atomic indices.

        Returns:
            dict: Updated input dictionary with properties.Rij
        """
        # atomic positions: N_atoms x 3
        positions = inputs[properties.R]
        Rij = positions[inputs[properties.idx_j]] - positions[inputs[properties.idx_i]]
        inputs[properties.Rij] = Rij.to(self.device)

        return inputs

    def _epoch_step(self, pbar, train, converter):
        """
        train == True -> train, False -> validation
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss = 0.0
        for batch in pbar:
            # Convert smiles to atoms
            atoms_list = []
            for smiles in batch['smiles']:
                atoms = self.smiles_to_atoms(smiles)
                if atoms is not None:
                    atoms_list.append(atoms)
                else:
                    print(f"Skipping SMILES: {smiles}")
                    continue

            if not atoms_list:
                print("No valid atoms found in batch, skipping...")
                continue
                
            batch_schnet_input = converter(atoms_list)
            batch_schnet_input = self.calculate_Rij(batch_schnet_input)
            protein_embedding = batch['protein_embedding'].to(self.device)
            docking_scores = batch['docking_score'].to(self.device)

            # Forward pass
            pred_scores = self.model(batch_schnet_input, protein_embedding)
            batch_loss = self.criterion(pred_scores, docking_scores)

            # Backward pass
            if train:
                self.optimizer.zero_grad()
                batch_loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Log
            loss += batch_loss.item()
            pbar.set_postfix({'loss': batch_loss.item() / len(batch)})
            if train:
                wandb.log({'train_batch_loss': batch_loss.item() / len(batch)})
            else:
                wandb.log({'val_batch_loss': batch_loss.item() / len(batch)})

        return loss
    
    def train(self, epochs, save_frequency, train_dataloader, val_dataloader, checkpoint_dir):
        converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=5.), dtype=torch.float32, device=self.device)
        for epoch in range(epochs):
            # Train
            print(f"Epoch {epoch+1}/{epochs}")
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            train_loss = self._epoch_step(pbar, train=True, converter=converter)

            # Validation
            pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            val_loss = self._epoch_step(pbar, train=False, converter=converter)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # WandB logging
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss / len(train_dataloader),
                'val_loss': val_loss / len(val_dataloader)
            })

            # Save checkpoint
            if (epoch+1) % save_frequency == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

def main():
    # Parse command line arguments
    args = parse_args()

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Device: {device}")
    print(f"GPU: {args.gpu}")

    # Load configurations
    file_config = load_config('filepath.yml')
    model_config = load_config(os.path.join(args.config_dir, args.model_config))
    train_config = load_config(os.path.join(args.config_dir, args.train_config))

    # Override parameters if specified
    if args.epochs is not None:
        train_config['epochs'] = args.epochs
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
    if args.lr is not None:
        train_config['lr'] = args.lr

    # Initialize WandB
    wandb.init(
        project='(schnet) Docking Score Prediction', 
        group=model_config['target'],
        name=f"target_{model_config['target']}_batch_{train_config['batch_size']}_lr_{train_config['lr']}",
        config={"train_config": train_config, "model_config": model_config}
        )
    
    # Set current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Directories
    model_dir = os.path.join(file_config['data']['model'], 'schnet_ds_regression')
    os.makedirs(os.path.join(model_dir, f"schnet_ds_{current_time}"), exist_ok=True)
    model_save_dir = os.path.join(model_dir, f"schnet_ds_{current_time}")

    # Save config files
    config_save_path = os.path.join(model_save_dir, 'model_config.pkl')
    with open(config_save_path, 'wb') as f:
        pickle.dump(model_config, f)
    print(f"Model config saved at {config_save_path}")

    # Data files
    train_file = os.path.join(file_config['data']['train'], train_config['train_file'])
    val_file = os.path.join(file_config['data']['val'], train_config['val_file'])

    # Train parameters
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    lr = float(train_config['lr'])
    save_frequency = train_config['save_frequency']

    # Dataloaders
    train_dataloader = get_dataloader(
        csv_file=train_file,
        smiles_max_len=model_config['smiles_max_len'],
        protein_max_len=model_config['protein_max_len'],
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = get_dataloader(
        csv_file=val_file,
        smiles_max_len=model_config['smiles_max_len'],
        protein_max_len=model_config['protein_max_len'],
        batch_size=16,
        shuffle=False
    )

    # Load model
    model = SchNetDockingScorePredictor(
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        ffn_hidden_dim=model_config['ffn_hidden_dim'],
        num_transformer_blocks=model_config['num_transformer_blocks'],
        schnet_n_interactions=model_config['schnet_n_interactions'],
        schnet_cutoff=model_config['schnet_cutoff'],
        dropout=model_config['dropout'],
        atoms_max_len=model_config['atoms_max_len'],
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    trainer = Trainer(model, optimizer, current_time, device, model_config['atoms_max_len'])
    trainer.train(epochs, save_frequency, train_dataloader, val_dataloader, model_save_dir)

    # Save final model
    model_path = os.path.join(model_save_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    wandb.save(model_path)

if __name__ == '__main__':
    main()
