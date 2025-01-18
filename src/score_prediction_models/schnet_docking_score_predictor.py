import torch
import torch.nn as nn
import numpy as np
from .transformer_block import TransformerBlock
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from schnetpack.representation import SchNet

class SchNetDockingScorePredictor(nn.Module):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            ffn_hidden_dim, 
            num_transformer_blocks, 
            schnet_n_interactions=3,
            schnet_cutoff=5.0,
            atoms_max_len=350,
            regressor_hidden_dim=512,
            dropout=0.1,
            return_attn_wts=False
        ):
        super(SchNetDockingScorePredictor, self).__init__()

        self.return_attn_wts = return_attn_wts

        self.embed_dim = embed_dim

        self.schnet = SchNet(
            n_atom_basis=embed_dim,
            n_interactions=schnet_n_interactions,
            cutoff=schnet_cutoff,
        )

        self.af2_embedding_dim_reducer = None
        if embed_dim != 385:
            self.af2_embedding_dim_reducer = nn.Sequential(
                nn.Linear(385, embed_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * atoms_max_len, regressor_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(regressor_hidden_dim, 1),
        )
        
    def smiles_to_atoms(self, smiles):
        """
        Convert SMILES to ASE Atoms object.
        
        Args:
        - smiles (str): SMILES string

        Returns:
        - ase.Atoms: ASE Atoms object
        """

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
        positions = np.array([list(conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))])

        # get atomic numbers
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

        # create ASE Atoms object
        atoms = Atoms(
            positions=positions,
            numbers=atomic_numbers
        )

        return atoms
    
    def forward(self, smiles_embedding, af2_embedding):
        """
        Forward pass for the SchNetDockingScorePredictor.
        """
        batch_atoms = [self.smiles_to_atoms(smiles) for smiles in smiles_embedding]

        # get SchNet embeddings
        schnet_embeddings = []
        for atoms in batch_atoms:
            schnet_embeddings.append(self.schnet(
                positions=torch.tensor(atoms.positions, dtype=torch.float32).unsqueeze(0),
                atomic_numbers=torch.tensor(atoms.numbers, dtype=torch.long).unsqueeze(0)
            ).unsqueeze(0))

        # cat embeddings
        schnet_embeddings = torch.nn.utils.rnn.pad_sequence(schnet_embeddings, batch_first=True)

        if self.af2_embedding_dim_reducer:
            af2_embedding = self.af2_embedding_dim_reducer(af2_embedding)
        
        # stack transformer blocks
        if self.return_attn_wts:
            self_attn_wts_list = []
            cross_attn_wts_list = []
            for block in self.transformer_blocks:
                schnet_embeddings, self_attn_wts, cross_attn_wts = block(schnet_embeddings, af2_embedding)
                self_attn_wts_list.append(self_attn_wts)
                cross_attn_wts_list.append(cross_attn_wts)
        else:
            for block in self.transformer_blocks:
                schnet_embeddings, _, _ = block(schnet_embeddings, af2_embedding)

        # flatten the embeddings
        flattened_embedding = schnet_embeddings.flatten(start_dim=1)

        docking_score = self.regressor(flattened_embedding)

        if self.return_attn_wts:
            return docking_score, self_attn_wts_list, cross_attn_wts_list
        
        return docking_score