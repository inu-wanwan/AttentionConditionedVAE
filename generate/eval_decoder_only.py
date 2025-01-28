import os
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
from scripts.utils import load_config, load_vae_model
from rdkit import Chem, DataStructs
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem, Draw
from src.data.dataloader import get_dataloader
from src.generation_models.decoder_only_model import DecoderOnlyCVAE
from src.score_prediction_models.docking_score_predictor import DockingScorePredictor
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

def get_embeddings_from_ligand_id(ligand_id):
    """
    Get SMILES and protein embeddings from the ligand ID.
    """
    # load config
    file_config = load_config('filepath.yml')
    data_config = load_config('data.yml')['drd3']

    test_dataloader = get_dataloader(
        csv_file=os.path.join(file_config['data']['test'], 'test_DRD3.csv'),
        smiles_vocab=pickle.load(open(file_config['vocab']['smiles'], 'rb')),
        protein_vocab=pickle.load(open(file_config['vocab']['protein'], 'rb')),
        smiles_max_len=data_config['smiles_max_len'],
        protein_max_len=data_config['protein_max_len'],
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for batch in test_dataloader:
        if batch['ligand_id'][0] == ligand_id:
            smiles_embedding = batch['smiles_embedding']
            protein_embedding = batch['protein_embedding']
            break

    return smiles_embedding, protein_embedding

def get_top_10_ligands_from_csv(csv_file):
    """
    Get top 10 ligands from the CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the top 10 ligands.

    Returns:
        list: List of top 10 ligand IDs.
    """
    df = pd.read_csv(csv_file)
    top_10_df = df.nlargest(10, 'Docking_score')
    top_10_ligands = top_10_df['Ligand_ID'].tolist()

    return top_10_ligands

def get_binwise_ligands_from_csv(csv_file):
    """
    Get ligands from the CSV file grouped by docking score bins.

    Args:
        csv_file (str): Path to the CSV file containing the ligands grouped by docking score bins.

    Returns:
        list: list of ligands sampled from each docking score bin.
    """
    df = pd.read_csv(csv_file)
    df['Docking_score_bin'] = pd.cut(df['Docking_score'], bins=10, labels=False)

    np.random.seed(42)

    binwise_ligands = []
    for i in range(10):
        bin_df = df[df['Docking_score_bin'] == i]
        bin_ligand = bin_df.sample(1)['Ligand_ID'].values[0]
        binwise_ligands.append(bin_ligand)
        
    return binwise_ligands

def calculate_canonical_smiles(smiles_list):
    """
    Converts a list of SMILES to their canonical form using RDKit.

    Args:
        smiles_list (list): List of SMILES strings.

    Returns:
        list: List of canonical SMILES strings.
    """
    canonical_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles.append(Chem.MolToSmiles(mol))
    return canonical_smiles


def calculate_diversity(fingerprints):
    """
    Calculates the diversity of a set of molecules using Tanimoto similarity.

    Args:
        fingerprints (list): List of molecular fingerprints.

    Returns:
        float: Diversity score (1 - average pairwise Tanimoto similarity).
    """
    n = len(fingerprints)
    if n < 2:
        return 0  # Not enough data for diversity calculation

    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(similarity)

    avg_similarity = np.mean(similarities) if similarities else 0
    diversity = 1 - avg_similarity
    return diversity


def calculate_metrics_with_moses(generated_data, smiles_column='SMILES'):
    """
    Calculates validity, uniqueness (after canonicalization), and diversity using the MOSES benchmark method.

    Args:
        generated_data (DataFrame): DataFrame containing generated compounds.
        smiles_column (str): Column in the DataFrame containing SMILES strings.

    Returns:
        dict: A dictionary containing the validity, uniqueness, and diversity metrics.
    """
    # Extract SMILES strings
    smiles_list = generated_data[smiles_column].tolist()

    # 1. Validity: Check if each SMILES string can be converted to a valid molecule
    valid_smiles = [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
    validity = len(valid_smiles) / len(smiles_list) if smiles_list else 0

    # 2. Canonicalize valid SMILES
    canonical_smiles = calculate_canonical_smiles(valid_smiles)

    # 3. Uniqueness: Calculate the ratio of unique canonical SMILES to all canonical SMILES
    unique_smiles = set(canonical_smiles)
    uniqueness = len(unique_smiles) / len(canonical_smiles) if canonical_smiles else 0

    # 4. Diversity: Calculate diversity using pairwise Tanimoto similarity
    mols = [Chem.MolFromSmiles(smiles) for smiles in unique_smiles]
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        for mol in mols if mol is not None
    ]
    diversity = calculate_diversity(fingerprints) if fingerprints else 0

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'diversity': diversity
    }

def parser_args():
    parser = argparse.ArgumentParser(description="Generate molecules using the trained decoder-only model.")
    parser.add_argument('--model_dir', '-m', type=str, required=True, help='Directory containing the trained model')
    parser.add_argument('--num_gen', '-n', type=int, default=100, help='Number of molecules to generate per ligand ID')
    parser.add_argument('--mode', '-md', type=str, default='top10', choices=['top10', 'binwise'], help='Mode to select ligands')
    parser.add_argument('--epoch', '-e', type=int, default=None, help='Epoch to use for evaluation')
    return parser.parse_args()

def main():
    """
    Main function to generate molecules using the trained decoder-only model.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse command line arguments
    args = parser_args()

    # Load config
    file_config = load_config('filepath.yml')

    # Directories
    model_dir = args.model_dir
    output_dir = file_config['data']['generated']
    test_dir = file_config['data']['test']

    # Load the trained model
    # Load config
    try:
        with open(os.path.join(model_dir, 'model_config.pkl'), 'rb') as f:
            model_config = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Model config file not found. Please check the model directory.")
    
    target = model_config['target']

    # Load the VAE model
    vae_model = load_vae_model().to(device)

    # Load the docking score prediction model
    docking_model_file = os.path.join(file_config['data']['docking'], model_config['docking_score_regression'], 'model.pth')
    docking_model_config = load_config(os.path.join(file_config['data']['docking'], model_config['docking_score_regression'], 'model_config.pkl'))

    try:
        with open(docking_model_config, 'rb') as f:
            docking_config = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Docking score prediction model config file not found. Please check the model directory.")
    
    if docking_config['target'] != model_config['target']:
        raise ValueError("Docking score prediction model target does not match the decoder-only model target.")
    
    docking_model = DockingScorePredictor(
        embed_dim=docking_config['embed_dim'],
        num_heads=docking_config['num_heads'],
        ffn_hidden_dim=docking_config['ffn_hidden_dim'],
        num_transformer_blocks=docking_config['num_transformer_blocks'],
        return_attn_wts=True,
    ).to(device)

    docking_model.load_state_dict(torch.load(docking_model_file))

    # Load the decoder-only model
    model = DecoderOnlyCVAE(
        smiles_vae=vae_model,
        docking_score_predictor=docking_model,
        af2_max_len=model_config['protein_max_len'],
        transformer_layer_used=model_config['transformer_layer_used'],
    ).to(device)

    if args.epoch is not None:
        model_file = os.path.join(model_dir, f"model_epoch_{args.epoch}.pth")
    else:
        model_file = os.path.join(model_dir, 'model.pth')

    model.load_state_dict(torch.load(model_file))

    # Get ligand IDs
    test_file = os.path.join(test_dir, f"test_{model_config['target']}.csv")

    if args.mode == 'top10':
        ligand_id_list = get_top_10_ligands_from_csv(test_file)
    else:
        ligand_id_list = get_binwise_ligands_from_csv(test_file)

    # Generate molecules
    generated_molecules = generate(model, device, ligand_id_list, args.num_gen)

    # Calculate metrics
    metrics = calculate_metrics_with_moses(generated_molecules, smiles_column='Generated_SMILES')

    # Save generated molecules
    output_file = os.path.join(output_dir, f"{target}", f"generated_molecules_{args.mode}.csv")
    test_df = pd.read_csv(test_file)

    result_df = pd.merge(test_df, generated_molecules, left_on='Ligand_ID', right_on='SEED_Ligand_ID')

    result_df.to_csv(output_file, index=False)
    print(f"Generated molecules saved to {output_file}")

    # Save metrics
    metrics_file = os.path.join(output_dir, f"{target}", f"metrics_{args.mode}.csv")
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")

if __name__ == '__main__':
    main()