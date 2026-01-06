import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.data import Data

# --- 1. 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. 导入模块 ---
from configs.model_config import (
    ITERATIVE_DRUG_ENCODER_CONFIG,
    MULTI_VIEW_RNA_ENCODER_CONFIG,
    RNA_MOTIF_ENCODER_CONFIG,
    MAIN_PREDICTOR_CONFIG,
    DRUG_RNA_SSIM_CONFIG,
    SSIM_LOSS_CONFIG,
    CROSS_ATTENTION_CONFIG,
    MODEL_VARIANT_CONFIG,
    AUX_LOSS_TYPE,
    TRAIN_CONFIG,
    PROCESSED_DATA_DIR,
    SELF_SUPERVISED_LEARNING_CONFIG # 导入SSL配置
)
from data_processing.utils import (
    NUCLEOTIDE_VOCAB_SIZE, 
    smiles_to_pyg_graph, 
    rna_to_pyg_multiview_data, 
    get_mol_descriptors
)
from data_processing.datasets import MoleculeWithFragmentsDataset
from models.main_model import AffinityModelMiliPyG
from utils.logger_utils import log_print
from utils.training_utils import (
    set_training_module_params, 
    set_seed, 
    train_one_epoch, 
    evaluate_model,
    get_normalization_params
)

# --- 3. 配置 ---
# 请确认这个路径是正确的
COLD_START_RAW_DIR = r"F:\code1\冷启动\冷启动1\sadta rna edge aatention\data\blind_test_with_structure"
SAVE_DIR = "cold_start_results"
os.makedirs(SAVE_DIR, exist_ok=True)

TASK_TYPE = 'regression' 
LOG_LEVEL_DEBUG = False
set_training_module_params(sys.stdout, LOG_LEVEL_DEBUG)

# ==================================================================================
#  Dataset 类修正
# ==================================================================================
class ColdStartDataset(MoleculeWithFragmentsDataset):
    def get_raw_dataframe(self):
        if self._raw_df is None:
            try:
                self._raw_df = pd.read_csv(self.raw_file_path, sep=',')
            except Exception as e:
                print(f"Error loading raw dataframe: {e}", file=sys.stderr)
                return None
        return self._raw_df

    def process(self):
        print(f"Processing raw data from: {self.raw_file_path} (Using COMMA separator)")
        try:
            df = pd.read_csv(self.raw_file_path, sep=',')
        except FileNotFoundError:
            print(f"Error: Raw data file not found at {self.raw_file_path}")
            return
        
        self._raw_df = df
        data_list = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Data"):
            drug_smi = row.get(self.drug_smiles_col)
            rna_seq = row.get(self.rna_seq_col)
            label_val = row.get(self.label_col)
            
            if not (isinstance(drug_smi, str) and drug_smi.strip() and 
                    isinstance(rna_seq, str) and rna_seq.strip() and 
                    pd.notna(label_val)):
                continue

            original_mol_graph = smiles_to_pyg_graph(drug_smi)
            rna_struct = row.get(self.rna_struct_col)
            rna_multiview_obj = rna_to_pyg_multiview_data(rna_seq, rna_struct)
            mol_descriptors_raw = get_mol_descriptors(drug_smi)

            if original_mol_graph is None or rna_multiview_obj is None or mol_descriptors_raw is None:
                continue

            pKd_value = float(label_val)
            cls_threshold = 4.0

            data_entry = Data(
                x=original_mol_graph.x,
                edge_index=original_mol_graph.edge_index,
                edge_attr=getattr(original_mol_graph, 'edge_attr', None),
                num_nodes=original_mol_graph.num_nodes,
                rna_data_obj=rna_multiview_obj,
                y_reg=torch.tensor([pKd_value], dtype=torch.float),
                y_cls=torch.tensor([1 if pKd_value >= cls_threshold else 0], dtype=torch.float),
                y=torch.tensor([pKd_value], dtype=torch.float),
                drug_smi_str=drug_smi,
                rna_sequence_str=rna_seq,
                rna_dot_bracket=rna_struct if rna_struct else '',
                mol_descriptors=mol_descriptors_raw
            )
            data_list.append(data_entry)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed data saved to {self.processed_paths[0]}")

# ==================================================================================

def get_dataset_for_file(filename):
    file_path = os.path.join(COLD_START_RAW_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    dataset = ColdStartDataset(
        root_dir=PROCESSED_DATA_DIR, 
        raw_file_path=os.path.abspath(file_path),
        drug_smiles_col='SMILES',
        rna_seq_col='Target_RNA_sequence',
        rna_struct_col='secondary_structure', 
        label_col='pKd'             
    )
    dataset.dataset_name = dataset.dataset_name + "_db_fix"
    return dataset

def run_cold_start_experiment():
    print("PROGRAM STARTED: Initializing Optimized Double-Blind setup...", flush=True)
    set_seed(100)
    
    scenario = 'rm' 
    
    log_print(f"\n{'#'*60}", "info")
    log_print(f"### STARTING DOUBLE-BLIND SCENARIO (OPTIMIZED) ###", "info")
    log_print(f"{'#'*60}\n", "info")

    datasets = []
    for i in range(1, 6):
        filename = f"cold_{scenario}{i}.csv"
        log_print(f"Loading dataset: {filename} ...", "info")
        try:
            ds = get_dataset_for_file(filename)
            _ = ds[0]
            datasets.append(ds)
        except Exception as e:
            log_print(f"Error loading {filename}: {e}", "error")
            return

    fold_results_pcc = []
    fold_results_scc = []
    fold_results_rmse = []

    header_line = f"{'Fold':<6} | {'Best Epoch':<12} | {'Val PCC':<14} | {'Val SCC':<14} | {'Val RMSE':<14}"
    log_print("=" * len(header_line), "info")
    log_print(header_line, "info")
    log_print("-" * len(header_line), "info")

    for fold_idx in range(5):
        test_dataset = datasets[fold_idx]
        raw_train_datasets_list = [datasets[j] for j in range(5) if j != fold_idx]
        
        # --- 双盲数据清洗 ---
        test_df = test_dataset._raw_df
        if test_df is None:
             test_df = pd.read_csv(test_dataset.raw_file_path, sep=',')

        forbidden_drugs = set(test_df['SMILES'].unique())
        forbidden_rnas = set(test_df['Target_RNA_sequence'].unique())
        
        log_print(f"Fold {fold_idx}: Test set has {len(forbidden_drugs)} unique Drugs and {len(forbidden_rnas)} unique RNAs.", "info")
        
        clean_train_subsets = []
        total_samples_dataset = 0
        total_clean_samples = 0
        
        for ds in raw_train_datasets_list:
            total_samples_dataset += len(ds)
            valid_indices = []
            
            for i in range(len(ds)):
                data_item = ds[i]
                smi = getattr(data_item, 'drug_smi_str', None)
                rna = getattr(data_item, 'rna_sequence_str', None)
                
                if rna is None:
                    raise RuntimeError("Data object missing 'rna_sequence_str'. Please DELETE the 'processed_data' folder and rerun!")

                if (smi not in forbidden_drugs) and (rna not in forbidden_rnas):
                    valid_indices.append(i)
            
            if len(valid_indices) > 0:
                clean_subset = Subset(ds, valid_indices)
                clean_train_subsets.append(clean_subset)
                total_clean_samples += len(clean_subset)
        
        if len(clean_train_subsets) > 0:
            train_dataset = ConcatDataset(clean_train_subsets)
        else:
            log_print(f"ERROR: Fold {fold_idx} - No training data left!", "error")
            continue
            
        log_print(f"  Double-Blind Filtering: {total_samples_dataset} -> {total_clean_samples} samples "
                  f"(Removed {total_samples_dataset - total_clean_samples}).", "warn")

        # --- 归一化 ---
        norm_params = None 
        try:
             norm_params = get_normalization_params(train_dataset)
        except:
             pass

        # --- DataLoader ---
        train_loader = PyGDataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, num_workers=0)
        test_loader = PyGDataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=0)

        # =========================================================
        # ⚡️ 调参区域：针对双盲(Double Blind)的极致优化 ⚡️
        # =========================================================
        
        # 1. 降低模型复杂度
        tuned_hidden_dim = 128       # 64 太小 -> 回调到 128
        tuned_dropout = 0.2          # 0.5 太狠 -> 回调到 0.2 (保留更多信息)
        
        # 2. 迭代次数保持较少 (防止过平滑)
        tuned_iterations = 5         # 保持在 4-5 层
        
        # 3. 优化器微调
        tuned_lr = 1e-4              # 保持不变
        tuned_weight_decay = 1e-5    # 1e-4 太强 -> 降到 1e-5
        
        # 4. 自监督权重 (适中)
        tuned_lambda_ssl = 0.3       # 0.5 可能干扰主任务 -> 降到 0.3

        # =========================================================

        # --- 配置重写 ---
        iter_config = ITERATIVE_DRUG_ENCODER_CONFIG.copy()
        iter_config['hidden_dim'] = tuned_hidden_dim
        iter_config['output_dim'] = tuned_hidden_dim
        iter_config['num_iterations'] = tuned_iterations
        iter_config['dropout_rate'] = tuned_dropout

        rna_view_config = MULTI_VIEW_RNA_ENCODER_CONFIG.copy()
        rna_view_config['view_seq_gnn_config'] = rna_view_config['view_seq_gnn_config'].copy()
        rna_view_config['view_pair_gnn_config'] = rna_view_config['view_pair_gnn_config'].copy()
        rna_view_config['view_seq_gnn_config']['hidden_dim'] = tuned_hidden_dim
        rna_view_config['view_seq_gnn_config']['output_node_dim'] = tuned_hidden_dim
        rna_view_config['view_pair_gnn_config']['hidden_dim'] = tuned_hidden_dim
        rna_view_config['view_pair_gnn_config']['output_node_dim'] = tuned_hidden_dim
        rna_view_config['fused_representation_dim'] = tuned_hidden_dim 
        rna_view_config['input_nucleotide_vocab_size'] = NUCLEOTIDE_VOCAB_SIZE

        rna_motif_config = RNA_MOTIF_ENCODER_CONFIG.copy()
        rna_motif_config['motif_embedding_dim'] = tuned_hidden_dim

        cross_attn_config = CROSS_ATTENTION_CONFIG.copy()
        cross_attn_config['hidden_dim'] = tuned_hidden_dim
        cross_attn_config['drug_dim'] = tuned_hidden_dim
        cross_attn_config['rna_dim'] = tuned_hidden_dim

        ssim_conf = DRUG_RNA_SSIM_CONFIG.copy()
        ssim_conf['drug_substruct_dim'] = tuned_hidden_dim
        ssim_conf['rna_substruct_dim'] = tuned_hidden_dim
        ssim_conf['projection_dim'] = tuned_hidden_dim

        predictor_config = MAIN_PREDICTOR_CONFIG.copy()
        current_ssim_out_dim = ssim_conf['output_dim']
        predictor_config['input_dim'] = tuned_hidden_dim * 3 + current_ssim_out_dim
        predictor_config['hidden_dim1'] = 256 
        predictor_config['hidden_dim2'] = 128
        predictor_config['dropout'] = tuned_dropout

        # 关键：更新自监督配置权重
        ssl_config_for_run = SELF_SUPERVISED_LEARNING_CONFIG.copy()
        ssl_config_for_run['lambda_ssl'] = tuned_lambda_ssl

        model_config = {
            'iterative_drug_config': iter_config,
            'multi_view_rna_encoder_config': rna_view_config,
            'rna_motif_encoder_config': rna_motif_config,
            'main_predictor_config': predictor_config,
            'ssim_config': ssim_conf,
            'ssim_loss_config': SSIM_LOSS_CONFIG,
            'cross_attention_config': cross_attn_config,
            'use_ssim': MODEL_VARIANT_CONFIG['enable_ssim_module'],
            'debug_mode': False,
            'aux_loss_type': AUX_LOSS_TYPE,
            'task': TASK_TYPE 
        }
        
        device = torch.device(TRAIN_CONFIG['device'])
        model = AffinityModelMiliPyG(**model_config).to(device)
        
        # 使用调整后的 LR 和 Weight Decay
        optimizer = optim.AdamW(model.parameters(), lr=tuned_lr, weight_decay=tuned_weight_decay)
        criterion = nn.MSELoss()

        # --- 训练循环 ---
        best_pcc = -1.0
        best_scc = -1.0
        best_rmse = float('inf')
        best_epoch = -1
        
        num_epochs = 1500
        patience = 500
        epochs_no_improve = 0
        
# 临时修改全局配置中的 lambda_ssl，以适应当前 epoch
        # 注意：这里我们直接利用 Python 引用特性修改字典
        SELF_SUPERVISED_LEARNING_CONFIG['lambda_ssl'] = tuned_lambda_ssl

        for epoch in range(1, num_epochs + 1):
            # 删除 override_ssl_config 参数，因为你的旧函数不支持
            train_loss, _, _, _, _, _ = train_one_epoch(
                model, train_loader, optimizer, device, criterion, epoch, 
                fold_idx + 1, TASK_TYPE, norm_params
            )
            
            val_loss, val_metrics, y_true, y_pred = evaluate_model(
                model, test_loader, device, criterion, epoch, 
                fold_idx + 1, TASK_TYPE, norm_params
            )
            
            current_pcc = val_metrics.get('val_pcc', 0)
            current_scc = val_metrics.get('val_scc', 0)
            current_rmse = val_metrics.get('val_rmse', float('inf'))
            
            if current_pcc > best_pcc:
                best_pcc = current_pcc
                best_scc = current_scc
                best_rmse = current_rmse
                best_epoch = epoch
                epochs_no_improve = 0 
            else:
                epochs_no_improve += 1
        
            if epochs_no_improve >= patience:
                log_print(f"Early stopping at epoch {epoch} (Best: {best_epoch})", "info")
                break
        
        log_print(
            f"{fold_idx + 1:<6} | {best_epoch:<12} | {best_pcc:.4f}         | {best_scc:.4f}         | {best_rmse:.4f}",
            "info"
        )

        fold_results_pcc.append(best_pcc)
        fold_results_scc.append(best_scc)
        fold_results_rmse.append(best_rmse)

    log_print("-" * len(header_line), "info")
    log_print(f"DOUBLE-BLIND SUMMARY:", "info")
    log_print(f"  PCC:  Mean={np.mean(fold_results_pcc):.4f} ± {np.std(fold_results_pcc):.4f}", "info")
    log_print(f"  SCC:  Mean={np.mean(fold_results_scc):.4f} ± {np.std(fold_results_scc):.4f}", "info")
    log_print(f"  RMSE: Mean={np.mean(fold_results_rmse):.4f} ± {np.std(fold_results_rmse):.4f}", "info")
    log_print("=" * len(header_line) + "\n", "info")

if __name__ == "__main__":
    try:
        run_cold_start_experiment()
    except Exception as e:
        log_print(f"Critical Error: {e}", "error")
        import traceback
        log_print(traceback.format_exc(), "error")