import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from tqdm import tqdm

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def load_smiles(smi_path):
    df = pd.DataFrame()
    with open(smi_path, "r") as f:
        df["SMILES"] = [line.strip() for line in f]
    
    return [Chem.MolFromSmiles(smile) for smile in tqdm(df["SMILES"], desc="Loading and Converting DMQP1M SMILES") if Chem.MolFromSmiles(smile) is not None]

def load_smiles_from_csv(csv_path, target, smiles_column="SMILES"):
    df = pd.read_csv(csv_path)
    df = df[df["Target"] == target]
    return [Chem.MolFromSmiles(smile) for smile in tqdm(df["SMILES"], desc="Loading an Converting SMILES") if Chem.MolFromSmiles(smile) is not None]

def compute_fingerprints(mols, radius=2, n_bits=2048):
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits) for m in mols]

def train_pca(fps, n_components=2, pca_path="data/model/pca/fitted_pca2d.pt"):
    print("Training PCA model...")
    pca = PCA(n_components=n_components)
    pca.fit(fps)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    return pca

def reduce_dimensionality(fps, pca):
    return pca.transform(fps)

def plot_chemical_space(dmqp1m_pca2d, active_pca2d, generated_pca2d, seed_pca2d, output_path):
    fig, ax = plt.subplots()
    
    data_sets = [dmqp1m_pca2d, active_pca2d]
    colors = ['Blues', 'Reds']  # ヒートマップのカラーマップ
    alpha = 0.5  # 透明度
    _min, _max = -3, 3  # 軸の範囲

    for data, color in zip(data_sets, colors):
        x = data[:, 0]
        y = data[:, 1]
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        X, Y = np.mgrid[_min:_max:100j, _min:_max:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        cs = ax.contourf(X, Y, Z, levels=8, cmap=color, alpha=alpha, extend='min')
        cs.cmap.set_under('white')
        cs.changed()
    
    # 生成化合物のプロット
    ax.scatter(generated_pca2d[:,0], generated_pca2d[:,1], c="blue", s=20, marker="o", label="generated")
    
    # シード化合物のプロット
    ax.scatter(seed_pca2d[:,0], seed_pca2d[:,1], c="red", s=50, marker="x", label="seed")
    
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)
    ax.legend()
    plt.savefig(output_path)

def load_active_smiles(dataset_csv, target):
    df = pd.read_csv(dataset_csv)
    df = df[df["Target"] == target]
    df = df[df["Active"] == True]
    return [Chem.MolFromSmiles(smile) for smile in tqdm(df["SMILES"], desc="Loading and Converting Active SMILES") if Chem.MolFromSmiles(smile) is not None]

def load_seed_smi(dataset_csv, seed_id):
    df = pd.read_csv(dataset_csv)
    df = df[df["Ligand_id"] == seed_id]
    return [Chem.MolFromSmiles(smile) for smile in df["SMILES"] if Chem.MolFromSmiles(smile) is not None]

def load_generated_smiles(generated_smiles, seed_id):
    df = pd.read_csv(generated_smiles)
    df = df[df["SEED_Ligand_ID"] == seed_id]
    return [Chem.MolFromSmiles(smile) for smile in tqdm(df["Generated_SMILES"], desc="Loading and Converting Generated SMILES") if Chem.MolFromSmiles(smile) is not None]

def main(dmqp1m_smi, dataset_csv, generated_smiles, pca_model, output_path, target, seed_id):
    print("Loading data...")
    dmqp1m_mols = load_smiles(dmqp1m_smi)
    active_mols = load_active_smiles(dataset_csv, target)
    generated_mols = load_generated_smiles(generated_smiles, seed_id)
    seed_mols = load_seed_smi(dataset_csv, seed_id)
    
    dmqp1m_fps = compute_fingerprints(dmqp1m_mols)
    active_fps = compute_fingerprints(active_mols)
    generated_fps = compute_fingerprints(generated_mols)
    seed_fps = compute_fingerprints(seed_mols)
    
    # PCA の学習またはロード
    if os.path.exists(pca_model):
        print("Loading PCA model...")
        pca = load_pickle(pca_model)
    else:
        pca = train_pca(dmqp1m_fps + active_fps)
    
    print("Reducing dimensions...")
    dmqp1m_pca2d = reduce_dimensionality(dmqp1m_fps, pca)
    active_pca2d = reduce_dimensionality(active_fps, pca)
    generated_pca2d = reduce_dimensionality(generated_fps, pca)
    seed_pca2d = reduce_dimensionality(seed_fps, pca)
    
    print("Plotting chemical space...")
    plot_chemical_space(dmqp1m_pca2d, active_pca2d, generated_pca2d, seed_pca2d, output_path)
    print("Plot saved at", output_path)

if __name__ == "__main__":
    target = "DRD3"
    seed_id = "CHEMBL57241"
    main(dmqp1m_smi="data/preprocessed/Druglike_million_canonical_no_dot_dup.smi", 
         dataset_csv="data/preprocessed/dataset_final_processed.csv", 
         generated_smiles=f"data/generated/{target}/generated_molecules_active.csv", 
         pca_model="data/model/pca/dmqp1m_pca2d.pt", 
         output_path="data/plots/decoder_only/2025-01-10_20-12-20/chemical_space.jpeg",
         target=target,
         seed_id=seed_id)
