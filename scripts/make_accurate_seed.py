import pandas as pd
import argparse
import os

def create_accuracy_sorted_subset(input_file, output_file, num_samples, ligand_file):
    """
    Create a subset of the input file sorted by accuracy and save it to the output file.
    """
    df = pd.read_csv(input_file)
    ligand_df = pd.read_csv(ligand_file)

    if num_samples > len(df):
        raise ValueError(f"Number of samples {num_samples} is greater than the number of rows in the input file {len(df)}")
    
    # Merge the input file with the ligand file to get the actual docking scores
    df = pd.merge(df, ligand_df[['Canonical_SMILES', 'Ligand_id']], on='Canonical_SMILES', how='left')
    
    df['Prediction_Error'] = abs(df['Actual_Docking_Score'] - df['Predicted_Docking_Score'])
    df_sorted = df.sort_values(by='Prediction_Error', ascending=True).head(num_samples)
    df_sorted.to_csv(output_file, index=False)
    print(f"Accuracy sorted subset saved to {output_file}")


def main():
    """
    Main function to create an accuracy sorted subset of the input file.
    """
    parser = argparse.ArgumentParser(description="Create an accuracy sorted subset of the input file.")
    parser.add_argument('-i', '--input_file', type=str, help='Path to the input file')
    parser.add_argument('-o', '--output_file', type=str, help='Path to save the accuracy sorted subset')
    parser.add_argument('-n', '--num_samples', type=int, help='Number of samples in the accuracy sorted subset')
    parser.add_argument('-l', '--ligand_file', type=str, help='Path to the ligand file')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    create_accuracy_sorted_subset(args.input_file, args.output_file, args.num_samples, args.ligand_file)

if __name__ == "__main__":
    main()