import torch
import os
import argparse
import pickle
import pandas as pd
import schnetpack as spk
from tqdm import tqdm
from utils import load_config, smiles_to_atoms, calculate_Rij
from src.data.dataloader import get_dataloader
from src.score_prediction_models.schnet_docking_score_predictor import SchNetDockingScorePredictor

def argparser():
    """
    Argument parser for command line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate the docking score prediction model.")
    parser.add_argument('--timestamp', '-t', type=str, required=True, help='Timestamp of the trained model')
    parser.add_argument('--use_epoch', '-e', type=int, default=None, help='Epoch to use for evaluation')
    return parser.parse_args()

def predict_scores(device, model, dataloader, output_file):
    """
    Predict docking scores using the model and dataloader, and save results to a CSV file.

    Args:
        device: torch.device, computation device (CPU or GPU).
        model: PyTorch model, the trained docking score predictor.
        dataloader: DataLoader, batched data loader for input data.
        output_file: str, path to save the results as a CSV file.
    """
    model.eval()
    model.to(device)
    
    converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=5.), dtype=torch.float32, device=device)

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting docking scores"):
            # Load batch data
            smiles = batch['smiles']
            protein_id = batch['protein_id']
            docking_scores = batch['docking_score'].to(device)
            smiles_embedding = batch['smiles_embedding'].to(device)
            protein_embedding = batch['protein_embedding'].to(device)

            # Convert smiles to atoms
            atoms_list = []
            for smiles in batch['smiles']:
                atoms = smiles_to_atoms(smiles)
                if atoms is not None:
                    atoms_list.append(atoms)
                else:
                    print(f"Skipping SMILES: {smiles}")
                    continue

            if not atoms_list:
                print("No valid atoms found in batch, skipping...")
                continue
                
            batch_schnet_input = converter(atoms_list)
            batch_schnet_input = calculate_Rij(batch_schnet_input, device)


            # Predict scores
            pred_scores = model(batch_schnet_input, protein_embedding).cpu()

            # Collect results
            for i in range(len(smiles)):
                results.append({
                    'SMILES': smiles[i],
                    'Protein_ID': protein_id[i],
                    'Actual_Docking_Score': docking_scores[i].item(),
                    'Predicted_Docking_Score': pred_scores[i].item(),
                    'Error': abs(docking_scores[i].item() - pred_scores[i].item())
                })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    """
    Main function to evaluate the docking score prediction model using batch processing.
    """
    # Parse command line arguments
    args = argparser()
    timestamp = args.timestamp
    use_epoch = args.use_epoch

    # Load configurations
    file_config = load_config('filepath.yml')
    model_config_file = os.path.join(file_config['data']['schnet'], f"schnet_ds_{timestamp}", 'model_config.pkl')

    with open(model_config_file, 'rb') as f:
        model_config = pickle.load(f)
    
    # File paths
    target = model_config['target']
    model_dir = file_config['data']['schnet']
    test_file = os.path.join(file_config['data']['test'], f"test_{target}.csv")
    os.makedirs(os.path.join(file_config['data']['eval'], 'schnet_ds_regression', timestamp), exist_ok=True)
    results_file = os.path.join(file_config['data']['eval'], 'schnet_ds_regression', timestamp, f"results_{target}.csv")

    if use_epoch is not None:
        model_file = os.path.join(model_dir, f"schnet_ds_{timestamp}", f"model_epoch_{use_epoch}.pth")
    else:
        model_file = os.path.join(model_dir, f"schnet_ds_{timestamp}", 'model.pth')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SchNetDockingScorePredictor(
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        ffn_hidden_dim=model_config['ffn_hidden_dim'],
        num_transformer_blocks=model_config['num_transformer_blocks'],
        schnet_n_interactions=model_config['schnet_n_interactions'],
        schnet_cutoff=model_config['schnet_cutoff'],
        atoms_max_len=model_config['atoms_max_len'],
        regressor_hidden_dim=model_config['regressor_hidden_dim'],
        dropout=model_config['dropout']
    )
    model.load_state_dict(torch.load(model_file))

    # DataLoader setup with batch processing
    batch_size = 128  # Define batch size
    test_dataloader = get_dataloader(
        csv_file=test_file,
        smiles_max_len=model_config['smiles_max_len'],
        protein_max_len=model_config['protein_max_len'],
        batch_size=batch_size,
        shuffle=False
    )

    # Predict docking scores and save results
    predict_scores(device, model, test_dataloader, results_file)

if __name__ == '__main__':
    main()
