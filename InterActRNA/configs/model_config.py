# drug_rna_mili_pyg/configs/model_config.py

import torch

# --- 模型类型选择 ---
DRUG_ENCODER_TYPE = 'iterative_gin'

# --- 原子特征相关 ---
ATOM_VOCAB_SIZE = 119
CHIRALITY_VOCAB_SIZE = 4

# --- 迭代药物编码器配置 (版本B: 大模型) ---
ITERATIVE_DRUG_ENCODER_CONFIG = {
    'node_emb_dim': 64,
    'hidden_dim': 256,
    'num_iterations': 15,
    'output_dim': 256,
    'atom_vocab_size': ATOM_VOCAB_SIZE,
    'chirality_vocab_size': CHIRALITY_VOCAB_SIZE,
    'dropout_rate': 0.3,
    'return_all_iterations': True
}

# --- RNA 编码器配置 (版本B: 大模型) ---
VIEW_GNN_CONFIG_TEMPLATE = {
    'hidden_dim': 256,
    'output_node_dim': 256,
    'num_layers': 2,
    'gnn_type': 'gat',
    'gat_heads': 4,
    'dropout': 0.2,
    'readout_pool_type': 'mean'
}
MULTI_VIEW_RNA_ENCODER_CONFIG = {
    'nucleotide_embedding_dim': 128,
    'view_seq_gnn_config': VIEW_GNN_CONFIG_TEMPLATE.copy(),
    'view_pair_gnn_config': VIEW_GNN_CONFIG_TEMPLATE.copy(),
    'fusion_method': 'concat',
    'fused_representation_dim': 256,
    'dropout_after_fusion': 0.2,
    'predict_node_scores_for_subgraph': True,
}

# --- RNA基序编码器配置 (版本B: 大模型) ---
RNA_MOTIF_ENCODER_CONFIG = {
    'motif_embedding_dim': 512,
    'num_heads': 4,
    'dropout': 0.2
}

# --- 子结构注意力配置 (动态关联) ---
SUBSTRUCTURE_ATTENTION_CONFIG = {
    'hidden_dim': ITERATIVE_DRUG_ENCODER_CONFIG['hidden_dim'],
    'num_iterations': ITERATIVE_DRUG_ENCODER_CONFIG['num_iterations'],
    'attention_dropout': 0.1
}

# --- 跨模态Cross-Attention配置 (版本B: 大模型) ---
CROSS_ATTENTION_CONFIG = {
    'drug_dim': ITERATIVE_DRUG_ENCODER_CONFIG['output_dim'],
    'rna_dim': MULTI_VIEW_RNA_ENCODER_CONFIG['fused_representation_dim'],
    'hidden_dim': 512,
    'num_heads': 4,
    'dropout': 0.1
}

# --- SSIM模块配置 (版本B: 大模型) ---
DRUG_RNA_SSIM_CONFIG = {
    'drug_substruct_dim': ITERATIVE_DRUG_ENCODER_CONFIG['hidden_dim'],
    'rna_substruct_dim': RNA_MOTIF_ENCODER_CONFIG['motif_embedding_dim'],
    'projection_dim': 512,
    'num_drug_substructs': ITERATIVE_DRUG_ENCODER_CONFIG['num_iterations'],
    'similarity_method': 'cosine',
    'aggregation_method': 'attention',
    'output_dim': 64,
    'dropout': 0.1
}

# --- RNA 子图处理配置 (保持不变) ---
RNA_SUBGRAPH_NODE_OUTPUT_DIM = 32
RNA_SUBGRAPH_FINAL_REPRESENTATION_DIM = 64
RNA_SUBGRAPH_GNN_CONFIG_TEMPLATE = { 'hidden_dim': 32, 'output_node_dim': 32, 'num_layers': 2, 'gnn_type': 'gat', 'gat_heads': 2, 'dropout': 0.1, 'readout_pool_type': 'mean', 'predict_node_scores': False }
RNA_SUBGRAPH_CROSS_ATTENTION_CONFIG = { 'seq_view_dim': 32, 'pair_view_dim': 32, 'attention_hidden_dim': 32, 'output_dim': 64, 'dropout': 0.1 }
RNA_SUBGRAPH_CONFIG = { 'top_k_nodes': 5, 'num_hops': 2, 'min_nodes_for_subgraph': 3, 'subgraph_seq_gnn_config': RNA_SUBGRAPH_GNN_CONFIG_TEMPLATE.copy(), 'subgraph_pair_gnn_config': RNA_SUBGRAPH_GNN_CONFIG_TEMPLATE.copy(), 'cross_attention_config': RNA_SUBGRAPH_CROSS_ATTENTION_CONFIG.copy(), 'final_subgraph_representation_dim': 64 }

# --- 主预测头配置 (版本B: 大模型) ---
USE_SSIM_MODULE = True

PREDICTOR_INPUT_DIM = (
    ITERATIVE_DRUG_ENCODER_CONFIG['output_dim'] +
    RNA_MOTIF_ENCODER_CONFIG['motif_embedding_dim'] +
    MULTI_VIEW_RNA_ENCODER_CONFIG['fused_representation_dim'] +
    (DRUG_RNA_SSIM_CONFIG['output_dim'] if USE_SSIM_MODULE else 0)
)

MAIN_PREDICTOR_CONFIG = {
    'hidden_dim1': 1024,
    'hidden_dim2': 512,
    'output_dim': 1,
    'dropout': 0.3,
    'input_dim': PREDICTOR_INPUT_DIM
}

# --- 辅助损失配置 ---
AUX_LOSS_TYPE = 'none'
LAMBDA_AUX = 0.0
LAMBDA_AUX_LARGE_MOL = 0.0


CLASSIFICATION_THRESHOLD = 4.0
# --- SSIM损失配置 ---
SSIM_LOSS_CONFIG = { 'use_ssim_loss': False, 'ssim_loss_weight': 0.1, 'similarity_target_threshold': 0.7, 'pKd_min_val_for_norm': 5.0, 'pKd_max_val_for_norm': 12.0 }

# --- 迭代学习相关参数 ---
ITERATIVE_LEARNING_CONFIG = { 'enable_substructure_attention': True, 'attention_temperature': 1.0, 'substructure_pooling': 'mean', 'enable_residual_connections': False, 'use_ssim_module': True }

# --- 训练参数 ---
TRAIN_CONFIG = {
    'analyze_fragments_on_start': False,
    'weight_decay': 1e-7,
    'batch_size': 32,
    'num_epochs': 4000,
    'device': 'cuda'  if torch.cuda.is_available() else 'cpu',
    'lr_scheduler_type': 'fixed',
    'learning_rate': 1e-4,
    'early_stop_metric': 'val_pcc',
    'n_splits_kfold': 5,
}
# --- 新增: 自监督学习配置 ---
SELF_SUPERVISED_LEARNING_CONFIG = {
    'enable': True,                # 总开关
    'mask_rate': 0.15,               # 屏蔽15%的节点
    'lambda_ssl': 0.2,               # 自监督损失的权重
    'mask_type': 'learnable_token'              # 屏蔽方式: 'zero' 或 'learnable_token'
}

# --- 数据路径 ---
RAW_DATA_FILE_PATH = "data/All_sf_dataset_v1_2D.csv"
PROCESSED_DATA_DIR = "./processed_data/"

# --- CSV/TSV 文件中的列名 ---
COLUMN_NAMES = { 'drug_smiles': 'SMILES', 'rna_sequence': 'Target_RNA_sequence', 'rna_structure': 'secondary_structure', 'label': 'pKd' }

# --- 调试和模型变体配置 ---
DEBUG_CONFIG = { 'log_substructure_weights': False, 'visualize_attention': False, 'save_intermediate_features': False, 'log_ssim_scores': True }
MODEL_VARIANT_CONFIG = { 'use_iterative_drug_encoder': True, 'use_chemical_fragments': False, 'enable_cross_attention': True, 'enable_mili_style_loss': False, 'enable_ssim_module': True }

# --- 验证配置完整性的函数 ---
def validate_config():
    pass # 暂时跳过验证以简化