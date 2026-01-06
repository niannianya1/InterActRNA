# drug_rna/models/main_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .guided_attention import GuidedCrossAttention
from torch_geometric.nn import global_mean_pool
from .iterative_drug_encoder import IterativeDrugEncoder
from .ssim_module import DrugRNASSIM
from .encoders import MultiViewRNAEncoder
from .rna_motif_encoder import RNAMotifEncoder
from utils.logger_utils import log_print
import sys
import os
from .iterative_drug_encoder import IterativeDrugEncoder, TargetAwareFusion

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_from_model = os.path.dirname(current_dir)
if project_root_from_model not in sys.path:
    sys.path.insert(0, project_root_from_model)

class MLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.size(0) == 0:
            return torch.empty(0, self.fc3.out_features, device=x.device)
        
        # BN层需要至少2个样本，如果只有1个则跳过BN
        if x.size(0) > 1:
            x = self.dropout(F.relu(self.bn1(self.fc1(x))))
            x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        else:
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            
        x = self.fc3(x)
        return x



class AffinityModelMiliPyG(nn.Module):
    def __init__(self,
                 iterative_drug_config,
                 multi_view_rna_encoder_config,
                 rna_motif_encoder_config,
                 main_predictor_config,
                 ssim_config=None,
                 ssim_loss_config=None,
                 cross_attention_config=None,
                 use_ssim=True,
                 debug_mode=False,
                 **kwargs):  # Added **kwargs to accept unused params from training_utils
        super().__init__()
        self.debug_mode = debug_mode
        self.use_ssim = use_ssim and (ssim_config is not None)
        self.use_guided_attention = cross_attention_config is not None

        # --- 药物编码器 ---
        iterative_drug_config_modified = iterative_drug_config.copy()
        iterative_drug_config_modified['return_all_iterations'] = self.use_ssim
        self.drug_encoder = IterativeDrugEncoder(**iterative_drug_config_modified)
        self.drug_output_dim = iterative_drug_config['output_dim']

        # --- RNA分层编码器 ---
        self.rna_base_encoder = MultiViewRNAEncoder(**multi_view_rna_encoder_config)
        self.base_rna_output_dim = self.rna_base_encoder.final_output_dim

        rna_motif_config_updated = rna_motif_encoder_config.copy()
        rna_motif_config_updated['node_input_dim'] = self.base_rna_output_dim
        self.rna_motif_encoder = RNAMotifEncoder(**rna_motif_config_updated)
        self.motif_rna_output_dim = rna_motif_config_updated['motif_embedding_dim']

        # --- 创新点：靶点感知融合模块 (替换了旧的 SubstructureAttention 功能) ---
        self.target_aware_fusion = TargetAwareFusion(
            drug_dim=iterative_drug_config['hidden_dim'],
            rna_dim=self.motif_rna_output_dim,
            num_heads=cross_attention_config.get('num_heads', 4),  # 复用超参数
            dropout=cross_attention_config.get('dropout', 0.1)
        )

        # --- SSIM模块 ---
        self.ssim_module = None
        self.ssim_output_dim = 0
        if self.use_ssim and ssim_config:
            self.ssim_module = DrugRNASSIM(**ssim_config)
            self.ssim_output_dim = ssim_config['output_dim']

        self.use_ssim_loss = self.use_ssim and ssim_loss_config and ssim_loss_config.get('use_ssim_loss', False)
        if self.use_ssim_loss:
            self.ssim_loss_weight = ssim_loss_config.get('ssim_loss_weight', 0.1)
            self.pKd_min = ssim_loss_config.get('pKd_min_val_for_norm', 5.0)
            self.pKd_max = ssim_loss_config.get('pKd_max_val_for_norm', 12.0)

        # --- Guided Cross-Attention ---
        if self.use_guided_attention:
            attn_hidden_dim = cross_attention_config.get('hidden_dim', 256)
            attn_heads = cross_attention_config.get('num_heads', 4)
            self.rna_to_drug_attention = GuidedCrossAttention(query_dim=iterative_drug_config['hidden_dim'],
                                                              key_dim=self.base_rna_output_dim,
                                                              hidden_dim=attn_hidden_dim, num_heads=attn_heads)
            self.drug_to_rna_attention = GuidedCrossAttention(query_dim=self.base_rna_output_dim,
                                                              key_dim=iterative_drug_config['hidden_dim'],
                                                              hidden_dim=attn_hidden_dim, num_heads=attn_heads)

        self.task = kwargs.get('task', 'regression')

        predictor_input_dim = main_predictor_config['input_dim']
        hidden_dim1 = main_predictor_config['hidden_dim1']
        hidden_dim2 = main_predictor_config['hidden_dim2']
        dropout = main_predictor_config['dropout']

        # 共享的MLP“身体”部分
        self.shared_predictor_body = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 回归任务的“插头”
        self.regression_head = nn.Linear(hidden_dim2, 1)

        # 分类任务的“插头”
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim2, 1)
            # 注意：我们不在模型里加Sigmoid，而是使用BCEWithLogitsLoss，这样数值更稳定
        )

        # --- 辅助任务: 描述符预测头 ---
        num_descriptors = 10
        self.descriptor_predictor = nn.Sequential(
            nn.Linear(self.drug_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_descriptors)
        )
        self.lambda_descriptors = 0.1
        # --- 新增: 自监督学习的恢复预测头 ---
        # 药物原子特征恢复头 (输入是GNN输出的节点维度，输出是初始嵌入维度)
        self.drug_node_recon_head = nn.Linear(
            iterative_drug_config['hidden_dim'],
            iterative_drug_config['node_emb_dim']
        )
        # RNA核苷酸特征恢复头
        self.rna_node_recon_head = nn.Linear(
            multi_view_rna_encoder_config['fused_representation_dim'],
            multi_view_rna_encoder_config['nucleotide_embedding_dim']
        )
                

    def _log_debug_model(self, message):
        if self.debug_mode:
            log_print(f"[MODEL] {message}", level="debug")

    def forward(self, original_mol_pyg_batch, rna_multiview_pyg_batch, rna_dot_bracket_strings, ssl_config=None):
        # --- 1. 获取原始的、未被屏蔽的初始嵌入 ---
        drug_x = original_mol_pyg_batch.x
        original_drug_atom_features = self.drug_encoder.atom_embedding(drug_x[:, 0])
        if self.drug_encoder.chirality_embedding is not None and drug_x.size(1) > 1:
            original_drug_chirality_features = self.drug_encoder.chirality_embedding(drug_x[:, 1])
            original_h_drug_embedded = torch.cat([original_drug_atom_features, original_drug_chirality_features],
                                                 dim=-1)
        else:
            original_h_drug_embedded = original_drug_atom_features

        _, original_h_rna_embedded, _ = self.rna_base_encoder(rna_multiview_pyg_batch, return_embedding=True)

        h_drug_embedded_to_mask = original_h_drug_embedded.clone()
        h_rna_embedded_to_mask = original_h_rna_embedded.clone()

        # --- 自监督学习 - 随机屏蔽 ---
        self.masked_drug_indices = None
        self.masked_rna_indices = None
        if self.training and ssl_config and ssl_config.get('enable', False):
            mask_rate = ssl_config.get('mask_rate', 0.15)
            num_drug_nodes = h_drug_embedded_to_mask.size(0)
            perm_drug = torch.randperm(num_drug_nodes, device=h_drug_embedded_to_mask.device)
            num_mask_nodes_drug = int(mask_rate * num_drug_nodes)
            if num_mask_nodes_drug > 0:
                self.masked_drug_indices = perm_drug[:num_mask_nodes_drug]
                h_drug_embedded_to_mask[self.masked_drug_indices] = 0.0

            num_rna_nodes = h_rna_embedded_to_mask.size(0)
            perm_rna = torch.randperm(num_rna_nodes, device=h_rna_embedded_to_mask.device)
            num_mask_nodes_rna = int(mask_rate * num_rna_nodes)
            if num_mask_nodes_rna > 0:
                self.masked_rna_indices = perm_rna[:num_mask_nodes_rna]
                h_rna_embedded_to_mask[self.masked_rna_indices] = 0.0

        h_drug = self.drug_encoder.initial_proj(h_drug_embedded_to_mask)
        _, h_rna, _ = self.rna_base_encoder(rna_multiview_pyg_batch, initial_embedding=h_rna_embedded_to_mask)

        # --- 2. 交错式引导编码 ---
        all_drug_substructures_for_ssim = []
        num_iterations = self.drug_encoder.num_iterations
        gin_layers = self.drug_encoder.gin_convs
        iterations_per_block = max(1, num_iterations // 2)
        for i in range(num_iterations):
            h_drug = gin_layers[i](h_drug, original_mol_pyg_batch.edge_index)
            h_drug = F.relu(self.drug_encoder.dropout(h_drug))
            if self.use_guided_attention and (i + 1) % iterations_per_block == 0:
                h_drug = self.rna_to_drug_attention(h_drug, original_mol_pyg_batch.batch, h_rna,
                                                    rna_multiview_pyg_batch.batch)
                h_rna = self.drug_to_rna_attention(h_rna, rna_multiview_pyg_batch.batch, h_drug,
                                                   original_mol_pyg_batch.batch)
            if self.use_ssim:
                # 在池化前使用专门的投影层，可能学习到更好的子结构表示
                projected_h_drug = self.drug_encoder.substruct_proj(h_drug)
                iter_substruct_repr = global_mean_pool(projected_h_drug, original_mol_pyg_batch.batch)
                all_drug_substructures_for_ssim.append(iter_substruct_repr)

        # --- 3. 最终表示生成 ---
        drug_node_repr = h_drug

        rna_motif_features, rna_motif_batch_idx = self.rna_motif_encoder(h_rna, rna_multiview_pyg_batch.batch,
                                                                         rna_dot_bracket_strings)
        if rna_motif_features.numel() > 0:
            rna_graph_repr_high_level = global_mean_pool(rna_motif_features, rna_motif_batch_idx)
        else:
            rna_graph_repr_high_level = torch.zeros(original_mol_pyg_batch.num_graphs, self.motif_rna_output_dim,
                                                    device=h_drug.device)

        if self.use_ssim and all_drug_substructures_for_ssim:
            drug_substructures_tensor = torch.stack(all_drug_substructures_for_ssim, dim=1)
            fused_graph_repr = self.target_aware_fusion(
                drug_substructures=drug_substructures_tensor,
                rna_context=rna_graph_repr_high_level
            )
            drug_graph_repr = self.drug_encoder.output_proj(fused_graph_repr)
        else:
            drug_graph_repr = self.drug_encoder.output_proj(
                global_mean_pool(drug_node_repr, original_mol_pyg_batch.batch))

        rna_graph_repr_low_level = global_mean_pool(h_rna, rna_multiview_pyg_batch.batch)
        ssim_features, similarity_matrix = None, None
        if self.use_ssim and self.ssim_module and 'drug_substructures_tensor' in locals():
            ssim_features, similarity_matrix = self.ssim_module(drug_substructures=drug_substructures_tensor,
                                                                rna_motifs=rna_motif_features,
                                                                rna_motifs_batch=rna_motif_batch_idx)
        else:
            ssim_features = torch.zeros(drug_graph_repr.size(0), 0, device=drug_graph_repr.device)

        # --- 4. 特征融合与预测 ---
        feature_list = [drug_graph_repr, rna_graph_repr_high_level, rna_graph_repr_low_level]
        if self.use_ssim and ssim_features is not None and ssim_features.size(-1) > 0:
            feature_list.append(ssim_features)
        final_combined_features = torch.cat(feature_list, dim=1)
        # 通过共享的预测头“身体”
        shared_output = self.shared_predictor_body(final_combined_features)

        # 根据任务选择最终的“插头”
        if self.task == 'regression':
            affinity_prediction = self.regression_head(shared_output)
        elif self.task == 'classification':
            # 分类任务输出的是logit（原始分数），而不是概率
            affinity_prediction = self.classification_head(shared_output)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        # --- 5. 辅助任务输出 ---
        descriptor_prediction = self.descriptor_predictor(drug_graph_repr)
        drug_recon = None
        rna_recon = None
        if self.masked_drug_indices is not None:
            masked_drug_node_repr = drug_node_repr[self.masked_drug_indices]
            drug_recon = self.drug_node_recon_head(masked_drug_node_repr)
        if self.masked_rna_indices is not None:
            masked_rna_node_repr = h_rna[self.masked_rna_indices]
            rna_recon = self.rna_node_recon_head(masked_rna_node_repr)

        return affinity_prediction, descriptor_prediction, similarity_matrix, drug_recon, rna_recon, original_h_drug_embedded, original_h_rna_embedded

    def compute_loss(self, affinity_prediction, descriptor_prediction, 
                 drug_recon, rna_recon,
                 true_labels, true_descriptors, 
                 original_drug_embedding, original_rna_embedding,
                 similarity_matrix=None, criterion_affinity=None, ssl_config=None):
    
        # 1. 主任务损失 (亲和力)
        pred_squeezed = affinity_prediction.squeeze(-1) if affinity_prediction.ndim > 1 else affinity_prediction
        labels_squeezed = true_labels.squeeze(-1) if true_labels.ndim > 1 else true_labels
        if criterion_affinity is None:
            criterion_affinity = F.mse_loss
        loss_affinity = criterion_affinity(pred_squeezed, labels_squeezed)

        # 2. SSIM辅助损失 (如果启用)
        aux_loss = torch.tensor(0.0, device=affinity_prediction.device)
        mean_similarity_for_metric = torch.tensor(float('nan'))
        if self.use_ssim_loss and similarity_matrix is not None and similarity_matrix.numel() > 0:
            try:
                mean_similarity = similarity_matrix.mean(dim=[1, 2])
                mean_similarity_for_metric = mean_similarity.mean()
                target_similarity = (labels_squeezed - self.pKd_min) / (self.pKd_max - self.pKd_min)
                target_similarity = torch.clamp(target_similarity, 0.0, 1.0)
                ssim_loss = F.mse_loss(mean_similarity, target_similarity.detach())
                aux_loss = self.ssim_loss_weight * ssim_loss
            except Exception as e:
                log_print(f"Error calculating SSIM aux loss: {e}", level="warn")
        
        # 3. 描述符预测损失
        loss_descriptors = F.mse_loss(descriptor_prediction, true_descriptors.float())
        
        # 4. 自监督重构损失 (只在训练时计算)
        loss_ssl = torch.tensor(0.0, device=affinity_prediction.device)
        if self.training and ssl_config and ssl_config.get('enable', False):
            loss_drug_recon = torch.tensor(0.0, device=affinity_prediction.device)
            # 确保所有必需的张量都存在
            if self.masked_drug_indices is not None and drug_recon is not None and original_drug_embedding is not None:
                true_drug_features = original_drug_embedding[self.masked_drug_indices]
                loss_drug_recon = F.mse_loss(drug_recon, true_drug_features.detach())

            loss_rna_recon = torch.tensor(0.0, device=affinity_prediction.device)
            # 确保所有必需的张量都存在
            if self.masked_rna_indices is not None and rna_recon is not None and original_rna_embedding is not None:
                true_rna_features = original_rna_embedding[self.masked_rna_indices]
                loss_rna_recon = F.mse_loss(rna_recon, true_rna_features.detach())
            
            loss_ssl = loss_drug_recon + loss_rna_recon

        # 5. 总损失
        lambda_ssl = ssl_config.get('lambda_ssl', 0.2) if ssl_config and ssl_config.get('enable', False) else 0.0
        total_loss = loss_affinity + aux_loss + self.lambda_descriptors * loss_descriptors + lambda_ssl * loss_ssl

        return total_loss, loss_affinity, aux_loss, loss_descriptors, loss_ssl, mean_similarity_for_metric