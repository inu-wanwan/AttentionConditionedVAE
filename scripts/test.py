from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_atoms(smiles):
    """
    SMILES文字列をASEのAtomsオブジェクトに変換する関数。
    
    Args:
        smiles (str): 分子を表現するSMILES文字列。

    Returns:
        Atoms: ASEのAtomsオブジェクト。
    """
    # RDKitでSMILESを分子オブジェクトに変換
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # 分子の3D座標を生成
    mol = Chem.AddHs(mol)  # 水素を追加
    success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if success != 0:
        raise ValueError(f"Failed to generate 3D coordinates for: {smiles}")
    
    AllChem.UFFOptimizeMolecule(mol)  # 3D座標を最適化

    # 3D座標を取得
    conformer = mol.GetConformer()
    positions = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    # 原子番号を取得
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # ASEのAtomsオブジェクトを作成
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    return atoms

if __name__ == '__main__':
    # テスト用のSMILES文字列
    smiles = 'COc1ccc([C@H]2[C@@H]3[NH+]=c4ccc(Cl)cc4=C3CCN2C(=O)Nc2ccc(F)c(Cl)c2)cc1'

    # SMILES文字列からAtomsオブジェクトを生成
    atoms = smiles_to_atoms(smiles)
    print(atoms)
    print(f'Number of atoms: {len(atoms)}')
    print(f'Atomic numbers: {atoms.numbers}')
    print(f'Positions: {atoms.positions}')