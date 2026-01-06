# drug_rna_mili_pyg/data_processing/datasets.py

import os
import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
from tqdm import tqdm
import pandas as pd
import warnings
from rdkit import Chem
import sys

from .utils import smiles_to_pyg_graph, rna_to_pyg_multiview_data, get_mol_descriptors
from configs.model_config import COLUMN_NAMES

class MoleculeWithFragmentsDataset(InMemoryDataset):
    def __init__(self, root_dir, raw_file_path,
                 drug_smiles_col=COLUMN_NAMES['drug_smiles'],
                 rna_seq_col=COLUMN_NAMES['rna_sequence'],
                 rna_struct_col=COLUMN_NAMES.get('rna_structure', None),
                 label_col=COLUMN_NAMES['label'],
                 transform=None, pre_transform=None, pre_filter=None, fragment_method=None): # fragment_method 变为可选
        
        self.raw_file_path = raw_file_path
        self.drug_smiles_col = drug_smiles_col
        self.rna_seq_col = rna_seq_col
        self.rna_struct_col = rna_struct_col
        self.label_col = label_col
        
        self.dataset_name = os.path.splitext(os.path.basename(raw_file_path))[0] + "_multiviewRNA_vFinalNormalized"
        
        self._raw_df = None
        self.descriptor_min_vals = None
        self.descriptor_max_vals = None

        super().__init__(root_dir, transform, pre_transform, pre_filter)
        
        try:
            data_and_slices, norm_params = torch.load(self.processed_paths[0])
            self.data, self.slices = data_and_slices
            self.descriptor_min_vals = norm_params['min']
            self.descriptor_max_vals = norm_params['max']
            print(f"Loaded processed data and descriptor normalization params from {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"Processed file not found at {self.processed_paths[0]}. process() will be called.")
        except Exception as e:
            print(f"Error loading processed data from {self.processed_paths[0]}: {e}. Consider re-processing.")

    def get_raw_dataframe(self):
        if self._raw_df is None:
            try:
                self._raw_df = pd.read_csv(self.raw_file_path, sep='\t')
            except Exception as e:
                print(f"Error loading raw dataframe: {e}", file=sys.stderr)
                return None
        return self._raw_df

    @property
    def raw_file_names(self):
        return [os.path.basename(self.raw_file_path)]

    @property
    def processed_file_names(self):
        return f'{self.dataset_name}_pyg_data_with_norm.pt'

    def download(self):
        pass

    def process(self):
        print(f"Processing raw data from: {self.raw_file_path} with descriptor normalization")
        df = pd.read_csv(self.raw_file_path, sep='\t')
        self._raw_df = df

        all_descriptors = []
        valid_indices = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Step 1/3: Pre-scanning for normalization"):
            drug_smi = row.get(self.drug_smiles_col)
            rna_seq = row.get(self.rna_seq_col)
            
            if not (isinstance(drug_smi, str) and drug_smi.strip() and isinstance(rna_seq, str) and rna_seq.strip()):
                continue
            
            mol_graph = smiles_to_pyg_graph(drug_smi)
            rna_obj = rna_to_pyg_multiview_data(rna_seq, row.get(self.rna_struct_col))
            mol_desc = get_mol_descriptors(drug_smi)

            if mol_graph is not None and rna_obj is not None and mol_desc is not None:
                all_descriptors.append(mol_desc)
                valid_indices.append(index)

        if not all_descriptors:
            raise RuntimeError("No valid molecule descriptors found in the entire dataset.")

        descriptors_tensor = torch.stack(all_descriptors)
        self.descriptor_min_vals = torch.min(descriptors_tensor, dim=0).values
        self.descriptor_max_vals = torch.max(descriptors_tensor, dim=0).values
        self.descriptor_max_vals[self.descriptor_max_vals == self.descriptor_min_vals] += 1e-6
        print("Descriptor normalization params (min/max) calculated.")

        data_list = []
        df_valid = df.loc[valid_indices]
        for index, row in tqdm(df_valid.iterrows(), total=df_valid.shape[0], desc="Step 2/3: Processing and Normalizing Data"):
            drug_smi = row[self.drug_smiles_col]
            rna_seq = row[self.rna_seq_col]
            label = float(row[self.label_col])
            rna_struct = row.get(self.rna_struct_col)

            original_mol_graph = smiles_to_pyg_graph(drug_smi)
            rna_multiview_obj = rna_to_pyg_multiview_data(rna_seq, rna_struct)
            mol_descriptors_raw = get_mol_descriptors(drug_smi)
            

            mol_descriptors_normalized = (mol_descriptors_raw - self.descriptor_min_vals) / (self.descriptor_max_vals - self.descriptor_min_vals)
            pKd_value = float(row[self.label_col])

            # --- 定义分类任务的阈值 ---
            CLASSIFICATION_THRESHOLD = 4.0  # 您可以根据需要修改这个值

            data_entry = Data(
                x=original_mol_graph.x,
                edge_index=original_mol_graph.edge_index,
                edge_attr=getattr(original_mol_graph, 'edge_attr', None),
                num_nodes=original_mol_graph.num_nodes,
                rna_data_obj=rna_multiview_obj,

                # --- 同时保存两种类型的标签 ---
                y_reg=torch.tensor([pKd_value], dtype=torch.float),
                y_cls=torch.tensor([1 if pKd_value >= CLASSIFICATION_THRESHOLD else 0], dtype=torch.float),

                # 我们保留 'y' 字段为回归值，以确保现有代码的最大兼容性
                y=torch.tensor([pKd_value], dtype=torch.float),

                drug_smi_str=drug_smi,
                rna_dot_bracket=rna_struct if rna_struct else '',
                mol_descriptors=mol_descriptors_normalized
            )
            data_list.append(data_entry)

        print(f"Processed {len(data_list)} valid entries.")

        if self.pre_filter is not None: data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None: data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        
        norm_params = {'min': self.descriptor_min_vals, 'max': self.descriptor_max_vals}
        torch.save(((data, slices), norm_params), self.processed_paths[0])
        print(f"Processed data and normalization params saved to {self.processed_paths[0]}")