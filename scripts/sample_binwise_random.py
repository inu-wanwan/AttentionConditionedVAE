import pandas as pd
import numpy as np
import argparse
import os
from utils import load_config

def sample_from_bins_with_error_distribution_binwise(data, actual_col, predicted_col, bins=10, random_state=None):
    """
    Samples one value from each bin of the specified column based on error distribution calculated binwise.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        actual_col (str): Column name for actual docking scores.
        predicted_col (str): Column name for predicted docking scores.
        bins (int): The number of bins to divide the data into.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame containing one sample from each bin based on binwise error distribution.
    """
    # Bin the data based on the actual scores
    data['bin'] = pd.cut(data[actual_col], bins=bins, labels=False)

    # Set random seed if provided
    np.random.seed(random_state)
    
    # Sample one value from each bin based on binwise error distribution
    def weighted_sample_binwise(group):
        group['error'] = abs(group[actual_col] - group[predicted_col])  # Calculate error within each bin
        weights = group['error'] / group['error'].sum()  # Calculate weights as normalized errors within the bin
        return group.sample(1, weights=weights)  # Sample based on weights

    sampled_data = data.groupby('bin').apply(weighted_sample_binwise).reset_index(drop=True)

    # Drop the temporary columns
    sampled_data = sampled_data.drop(columns=['bin', 'error'])
    
    return sampled_data

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model' , '-m', type=str, help='Model name e.g. schnet_ds_regression, ds_regression, etc.')
    parser.add_argument('--predictor_timestamp', '-p', type=str, required=True, help='Timestamp of the trained prediction model')
    parser.add_argument('--vae_timestamp', '-v', type=str, required=True, help='Timestamp of the trained VAE model')
    parser.add_argument('--result_file', '-r', type=str, default='results_test.csv', help='Name of the result file')
    args = parser.parse_args()

    # Load the config file
    config = load_config("filepath.yml")
    eval_dir = config['data']['eval']
    result_dir = os.path.join(eval_dir, args.model, args.predictor_timestamp)
    result_file = os.path.join(result_dir, args.result_file)
    output_dir = os.path.join(eval_dir, 'ds_regression', args.predictor_timestamp)

    # Load the prediction results
    results = pd.read_csv(result_file)

    # Sample from bins based on error distribution
    sampled_results = sample_from_bins_with_error_distribution_binwise(results, 'Actual_Docking_Score', 'Predicted_Docking_Score', bins=10, random_state=42)

    # Save the sampled results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'binwise_random.csv')
    sampled_results.to_csv(output_file, index=False)

    print(f"Sampled results saved to {output_file}")

if __name__ == "__main__":
    main()