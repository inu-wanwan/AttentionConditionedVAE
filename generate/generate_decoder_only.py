import os
import torch
import pandas as pd
from scripts.utils import load_config, load_vae_model, load_docking_score_predictor
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from src.data.dataloader import get_dataloader
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor
from src.generation_models.moses_vae import SmilesVAE
from src.generation_models.decoder_only_model import DecoderOnlyCVAE
from tqdm import tqdm

def generate(model, device, ligand_id_list, num_gen):
    """
    Generate molecules using the trained decoder only model and associate them with their seed ligand IDs.

    Args:
        model: Trained decoder-only model.
        device: Device to run the model on ('cuda' or 'cpu').
        ligand_id_list (list): List of seed ligand IDs.
        num_gen (int): Number of molecules to generate per ligand ID.

    Returns:
        pd.DataFrame: DataFrame containing generated molecules and their corresponding seed ligand IDs.
    """
    results = []

    for ligand_id in tqdm(ligand_id_list, desc="Generating molecules"):
        # Generate molecules
        smiles_emb, protein_emb = get_embeddings_from_ligand_id(ligand_id, device)

        smiles_emb = smiles_emb.to(device)
        protein_emb = protein_emb.to(device)

        generated_molecules = model.generate(
            batch_size=num_gen,
            smiles_embedding=smiles_emb,
            af2_embedding=protein_emb,
        )

        for mol in generated_molecules:
            results.append({
                'SEED_Ligand_ID': ligand_id,
                'Generated_SMILES': mol
            })
    
    results_df = pd.DataFrame(results)

    return results_df
    
def get_embeddings_from_ligand_id(ligand_id, device):
    """
    Get SMILES and protein embeddings from the ligand ID.
    """
    # load config
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')['drd3']

    test_dataloader = get_dataloader(
        csv_file=os.path.join(file_config['data']['test'], 'test_DRD3.csv'),
        smiles_max_len=data_config['smiles_max_len'],
        protein_max_len=data_config['protein_max_len'],
        batch_size=1,
        shuffle=False,
        ligand_id=ligand_id
    )

    data = next(iter(test_dataloader))

    smiles_emb = data['smiles_embedding']
    protein_emb = data['protein_embedding']

    return smiles_emb, protein_emb

def main():
    """
    Main function to generate molecules using the trained decoder only model.
    """
    # Load configurations
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')['drd3']
    model_config = load_config('model.yml')['decoder_only_model']

    # set config
    model_timestamp = '2025-01-10_20-12-20'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # paths
    model_path = os.path.join(file_config['data']['decoder_only'], model_timestamp, 'checkpoint', 'model_epoch_17.pth')
    ligand_file = os.path.join(file_config['data']['test'], 'accurate_10_seed.csv')
    out_dir = os.path.join(file_config['data']['eval'], 'decoder_only', model_timestamp)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'accurate.csv')

    # Load VAE model
    vae_model = load_vae_model().to(device)

    # Load docking score prediction model
    docking_model = load_docking_score_predictor("2025-01-03_22-35-27").to(device)

    # Load decoder only model
    model = DecoderOnlyCVAE(
        smiles_vae=vae_model,
        docking_score_predictor=docking_model,
        af2_max_len=model_config['af2_max_len'],
    ).to(device)

    # Load trained model
    model.load_state_dict(torch.load(model_path))

    # Generate molecules
    ligand_id_list = pd.read_csv(ligand_file)['Ligand_id'].tolist()
    generated_molecules_df = generate(model, device, ligand_id_list, num_gen=100)

    # Save generated molecules
    generated_molecules_df.to_csv(out_file, index=False)
    print(f"Generated molecules saved to {out_file}")

if __name__ == '__main__':
    main()