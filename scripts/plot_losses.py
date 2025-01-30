import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os

def plot_losses(train_loss_df, val_loss_df, save_path):
    """
    Plot the training and validation losses.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training losses.
        val_df (pd.DataFrame): DataFrame containing the validation losses.
        save_path (str): Path to save the plot.
    """
    plt.rcParams.update({'font.size': 18}) 
    epochs = range(1, len(train_loss_df) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_df["train_loss"], label="Train Loss", color='blue')  
    plt.plot(epochs, val_loss_df["val_loss"], label="Validation Loss", color='red')
    plt.ylim(0, 0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(save_path, dpi=600)
    print(f"Plot saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation losses.")
    parser.add_argument("--train_loss", "-t", type=str, required=True, help="Path to the training loss CSV file.")
    parser.add_argument("--val_loss", "-v", type=str, required=True, help="Path to the validation loss CSV file.")
    parser.add_argument("--save_path", "-s", type=str, required=True, help="Path to save the plot.")
    args = parser.parse_args()

    train_loss_df = pd.read_csv(args.train_loss)
    val_loss_df = pd.read_csv(args.val_loss)
    plot_losses(train_loss_df, val_loss_df, args.save_path)