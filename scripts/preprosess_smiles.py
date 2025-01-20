from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from utils import load_config
import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm

def extract_higher_MW(s_list: list) -> list:
    out = []
    for s in s_list:
        smis = s.split('.')
        mols = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)

        if len(mols) == 0:
            continue

        Mws = [Descriptors.MolWt(mol) for mol in mols]
        max_Mw_idx = Mws.index(max(Mws))

        # 分子量600以上のものを排除
        if Mws[max_Mw_idx] > 600:
            continue
        
        out.append(Chem.MolToSmiles(mols[max_Mw_idx]))
    return out

def remove_dup(smis: list) -> list:
    smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smis]
    smis = list(set(smis))
    return smis

def process_and_filter_smiles(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    tqdm.pandas(desc="Processing SMILES")

    # smiles列を処理し、有効な結果のみを保持
    def process_smiles(smiles):
        processed = remove_dup(extract_higher_MW([smiles]))
        return ".".join(processed) if processed else None

    df[smiles_col] = df[smiles_col].progress_apply(process_smiles)
    
    # smiles列がNoneまたは空の行を削除
    df = df[df[smiles_col].notna() & (df[smiles_col] != "")]
    
    return df

def main():
    config = load_config("filepath.yml")
    preprocessed_dir = config['data']['preprocessed']

    # Load dataset
    df = pd.read_csv(os.path.join(preprocessed_dir, 'dataset_final.csv'))

    # Process and filter SMILES
    df = process_and_filter_smiles(df, "Canonical_SMILES")

    # Save the processed dataset
    df.to_csv(os.path.join(preprocessed_dir, 'dataset_final_processed.csv'), index=False)
    print("Processed dataset saved at", os.path.join(preprocessed_dir, 'dataset_final_processed.csv'))

if __name__ == "__main__":
    main()