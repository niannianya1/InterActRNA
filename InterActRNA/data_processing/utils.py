# drug_rna_mili_pyg/data_processing/utils.py

import torch
from torch_geometric.data import Data
from rdkit import Chem
# 移除 3D 相关导入，因为我们回到 2D GIN
# from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors # 分子描述符
from rdkit.Chem import BRICS, Recap # BRICS 和 Recap 碎片化方法



# --- 原子特征相关常量 ---
ATOM_VOCAB_SIZE_UTIL = 119
CHIRALITY_VOCAB_SIZE_UTIL = 4
CHIRALITY_LIST = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]

def get_atom_features(atom):
    """
    获取单个原子的特征。
    返回原子类型索引和手性标签索引。
    """
    # 1. 提取原子序数
    try:
        atom_type_idx = atom.GetAtomicNum()
        if atom_type_idx >= ATOM_VOCAB_SIZE_UTIL:
            atom_type_idx = 0
    except:
        atom_type_idx = 0

    try:
        chiral_tag_idx = CHIRALITY_LIST.index(str(atom.GetChiralTag()))
    except ValueError:
        chiral_tag_idx = CHIRALITY_LIST.index("CHI_UNSPECIFIED")

    return [atom_type_idx, chiral_tag_idx]


def smiles_to_pyg_graph(smiles_string):
    """
    将 SMILES 字符串转换为 PyG Data 对象，只生成 2D 图信息。
    如果 SMILES 无效，则返回 None。
    """
    # 步骤 1: 解析SMILES
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    # 步骤 2: 提取所有节点的特征
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_features(atom))

    if not atom_features_list:
        return None

    x = torch.tensor(atom_features_list, dtype=torch.long)
    # 步骤 3: 提取所有边的连接关系
    edge_indices = []
    if mol.GetNumBonds() > 0:
        for bond in mol.GetBonds():
            # 获取键两端原子的索引
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # 添加双向边
            edge_indices.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
    # 返回 PyG Data 对象，不包含 pos
    return Data(x=x, edge_index=edge_index)


def get_molecule_fragments(smiles_string, method='brics'):
    """
    模仿 MILI 的 chemistryProcess.py 中的 get_substructure 函数核心逻辑，
    用于获取分子的化学碎片。
    """
    if not isinstance(smiles_string, str) or not smiles_string.strip():
        return set()

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return set()

    substructures = set()

    try:
        if method.lower() == 'brics':
            substructures = BRICS.BRICSDecompose(mol)
        elif method.lower() == 'recap':
            recap_tree = Recap.RecapDecompose(mol)
            leaves = recap_tree.GetLeaves()
            substructures = set(leaves.keys())
        else:
            return set()
    except Exception as e:
        return set()

    return {s for s in substructures if isinstance(s, str)}


def get_mol_descriptors(smiles_string):
    """
    提取 RDKit 分子描述符。
    选择一些常用的描述符，它们对分子的基本性质有很好的概括。
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    try:
        descriptors = [
            Descriptors.MolWt(mol),            # 分子量
            Descriptors.MolLogP(mol),          # LogP（亲脂性）
            Descriptors.TPSA(mol),             # 拓扑极性表面积
            Descriptors.NumHDonors(mol),       # 氢键供体数量
            Descriptors.NumHAcceptors(mol),    # 氢键受体数量
            Descriptors.NumRotatableBonds(mol),# 可旋转键数量
            Descriptors.NumAromaticRings(mol), # 芳香环数量
            Descriptors.NumSaturatedRings(mol),# 饱和环数量
            Descriptors.NumHeteroatoms(mol),   # 杂原子数量
            Descriptors.HeavyAtomCount(mol)    # 重原子数量 (非氢原子数量)
        ]

        clean_descriptors = [0.0 if (isinstance(d, float) and (d == float('nan') or d == float('inf'))) else d for d in descriptors]

        return torch.tensor(clean_descriptors, dtype=torch.float)
    except Exception as e:
        return None


# --- RNA 相关的函数 ---
NUCLEOTIDE_DICT = {'A': 0, 'U': 1, 'C': 2, 'G': 3, 'N': 4, 'T': 1}
NUCLEOTIDE_VOCAB_SIZE = len(set(NUCLEOTIDE_DICT.values()))

def rna_sequence_to_indices(sequence):
    """
    将 RNA 序列转换为核苷酸索引列表。
    """
    if not sequence: return []
    return [NUCLEOTIDE_DICT.get(n.upper(), NUCLEOTIDE_DICT['N']) for n in sequence]


def parse_dot_bracket_to_pairs(dot_bracket_string):
    """
    解析点括号表示的 RNA 二级结构，生成配对的碱基索引。
    """
    stack, pairs = [], []
    for i, char in enumerate(dot_bracket_string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append(tuple(sorted((i, j))))
    return pairs


def rna_to_pyg_multiview_data(sequence, structure_dot_bracket=None):
    """rna 转图
    将 RNA 序列和二级结构转换为 PyG Data 对象的多视图表示。
    包含序列连接边 (edge_index_seq) 和配对边 (edge_index_pair)。
    """
    if not sequence: return None
    # 步骤 1: 提取节点特征
    node_indices = rna_sequence_to_indices(sequence)
    if not node_indices: return None

    x_rna = torch.tensor(node_indices, dtype=torch.long).unsqueeze(-1)
    num_nodes_rna = x_rna.size(0)
    if num_nodes_rna == 0: return None
    # 步骤 2: 构建序列边
    edge_list_seq = []
    if num_nodes_rna > 1:
        for i in range(num_nodes_rna - 1):
            edge_list_seq.append((i, i + 1));
            edge_list_seq.append((i + 1, i))
    edge_index_seq = torch.tensor(edge_list_seq, dtype=torch.long).t().contiguous() if edge_list_seq else torch.empty((2, 0), dtype=torch.long)
    if edge_list_seq: edge_index_seq = torch.unique(edge_index_seq, dim=1)
    # 步骤 3: 构建结构边
    edge_list_pair = []
    if structure_dot_bracket and len(structure_dot_bracket) == len(sequence):
        try:
            base_pairs = parse_dot_bracket_to_pairs(structure_dot_bracket)
            for i, j in base_pairs:
                edge_list_pair.append((i, j));
                edge_list_pair.append((j, i))
        except Exception as e:
            pass
    edge_index_pair = torch.tensor(edge_list_pair, dtype=torch.long).t().contiguous() if edge_list_pair else torch.empty((2, 0), dtype=torch.long)
    if edge_list_pair: edge_index_pair = torch.unique(edge_index_pair, dim=1)

    return Data(x=x_rna, edge_index_seq=edge_index_seq, edge_index_pair=edge_index_pair, num_nodes=num_nodes_rna)

    # --- 新增：RNA二级结构基元解析 ---

def parse_rna_motifs_simple(dot_bracket_string):
    """
    一个简化的RNA二级结构解析器，用于从点括号表示法中区分茎区(stem)和环区(loop)。

    Args:
        dot_bracket_string (str): RNA的二级结构，例如 "((...))".

    Returns:
        dict: 一个字典，包含两类基元的节点索引列表。
              例如: {'stem': [0, 1, 5, 6], 'loop': [2, 3, 4]}
              如果某类基元不存在，则对应的列表为空。
    """
    if not isinstance(dot_bracket_string, str) or not dot_bracket_string:
        return {'stem': [], 'loop': []}

    n = len(dot_bracket_string)
    stack = []
    pairs = []

    # 1. 找到所有配对的碱基
    for i, char in enumerate(dot_bracket_string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                # 存储配对 (j, i)
                pairs.append(tuple(sorted((i, j))))

    # 2. 识别所有配对和未配对的索引
    paired_indices = set()
    for i, j in pairs:
        paired_indices.add(i)
        paired_indices.add(j)

    unpaired_indices = set(range(n)) - paired_indices

    return {
        'stem': sorted(list(paired_indices)),
        'loop': sorted(list(unpaired_indices))
    }