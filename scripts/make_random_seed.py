import pandas as pd
import argparse
import os

def create_ramdom_subset(input_file, output_file, num_samples):
    """
    Create a random subset of the input file and save it to the output file.
    """
    df = pd.read_csv(input_file)

    if num_samples > len(df):
        raise ValueError(f"Number of samples {num_samples} is greater than the number of rows in the input file {len(df)}")
    
    df_sample = df.sample(n=num_samples, random_state=42)
    df_sample.to_csv(output_file, index=False)
    print(f"Random subset saved to {output_file}")

def main():
    """
    Main function to create a random subset of the input file.
    """
    parser = argparse.ArgumentParser(description="Create a random subset of the input file.")
    parser.add_argument('-i', '--input_file', type=str, help='Path to the input file')
    parser.add_argument('-o', '--output_file', type=str, help='Path to save the random subset')
    parser.add_argument('-n', '--num_samples', type=int, help='Number of samples in the random subset')
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    create_ramdom_subset(args.input_file, args.output_file, args.num_samples)

if __name__ == "__main__":
    main()