# models/ssim_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from utils.logger_utils import log_print

# 导入新的RNA基元解析工具
from data_processing.utils import parse_rna_motifs_simple
from utils.logger_utils import log_print


class DrugRNASSIM(nn.Module):
    """
    Drug-RNA Substructure Similarity Interaction Module (SSIM)
    (V2 - Structural Motifs)
    基于SA-DDI的SSIM思想，并改进用于药物-RNA亲和力预测:
    1. 提取药物的多半径子结构（来自迭代编码器）
    2. [新] 显式地从RNA二级结构中提取结构基元（茎区、环区）作为子结构
    3. 计算药物-RNA子结构间的相似性
    4. 聚合得到全局相似性特征
    """

    def __init__(self,
                 drug_substruct_dim=128,
                 rna_substruct_dim=128,
                 projection_dim=128,
                 num_drug_substructs=10,

                 similarity_method='cosine',
                 aggregation_method='attention',
                 output_dim=64,
                 dropout=0.1,
                 debug_mode=False):
        super().__init__()

        self.debug_mode = debug_mode
        self.drug_substruct_dim = drug_substruct_dim
        self.rna_substruct_dim = rna_substruct_dim # 这是高层基序的维度

        self.projection_dim = projection_dim
        self.num_drug_substructs = num_drug_substructs

        self.similarity_method = similarity_method
        self.aggregation_method = aggregation_method
        self.output_dim = output_dim

        # 特征投影到统一维度
        self.drug_proj = nn.Linear(drug_substruct_dim, projection_dim)
        self.rna_proj = nn.Linear(rna_substruct_dim, projection_dim)

        # 不同的相似性计算方法 (这部分逻辑保持您原有的不变)
        if similarity_method == 'mlp':
            self.similarity_mlp = nn.Sequential(
                nn.Linear(projection_dim * 2, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim // 2),
                nn.ReLU(),
                nn.Linear(projection_dim // 2, 1)
            )
        elif similarity_method == 'bilinear':
            self.bilinear = nn.Bilinear(projection_dim, projection_dim, 1)

        # 聚合方法相关的参数 (这部分逻辑保持您原有的不变)
        if aggregation_method == 'attention':
            # --- 核心修复：移除 batch_first=True ---
            self.drug_attention = nn.MultiheadAttention(
                embed_dim=projection_dim, num_heads=4, dropout=dropout # batch_first=True 已移除
            )
            self.rna_attention = nn.MultiheadAttention(
                embed_dim=projection_dim, num_heads=4, dropout=dropout # batch_first=True 已移除
            )
            self.drug_query_proj = nn.Linear(projection_dim, projection_dim)
            self.rna_query_proj = nn.Linear(projection_dim, projection_dim)

        # 最终输出层 (这部分逻辑保持您原有的不变)
        final_input_dim = self._calculate_final_input_dim()
        self.final_proj = nn.Sequential(
            nn.Linear(final_input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
    def _compute_similarity_stats(self, similarity_matrix):
        """计算相似性矩阵的统计特征 (V-SIMPLE)"""
        # 输入的 similarity_matrix 现在是 [N_drug, N_rna]，是一个样本内部的矩阵
        if similarity_matrix.numel() == 0:
            # 如果没有RNA基元，返回4个0
            return torch.zeros(4, device=similarity_matrix.device)
            
        max_sim = similarity_matrix.max()
        mean_sim = similarity_matrix.mean()
        std_sim = similarity_matrix.std()
        # median需要对扁平化的张量操作
        median_sim = torch.median(similarity_matrix.flatten())

        return torch.stack([max_sim, mean_sim, std_sim, median_sim])    

    def _calculate_final_input_dim(self):
        """计算最终投影层的输入维度"""
        base_dim = 0

        if self.aggregation_method == 'attention':
            base_dim += self.projection_dim * 2  # 药物和RNA的attention输出
        else:
            base_dim += self.projection_dim * 2  # 药物和RNA的聚合特征

        # 添加全局统计特征
        base_dim += 4  # max_sim, mean_sim, std_sim, median_sim

        return base_dim

    

    def compute_pairwise_similarity(self, drug_feats, rna_feats):
        """
        计算药物-RNA子结构间的成对相似性

        Args:
            drug_feats: [batch_size, num_drug_sub, projection_dim]
            rna_feats: [batch_size, num_rna_sub, projection_dim]

        Returns:
            similarity_matrix: [batch_size, num_drug_sub, num_rna_sub]
        """
        batch_size, num_drug, _ = drug_feats.shape
        _, num_rna, _ = rna_feats.shape

        if self.similarity_method == 'cosine':
            drug_norm = F.normalize(drug_feats, dim=-1)
            rna_norm = F.normalize(rna_feats, dim=-1)
            similarity_matrix = torch.bmm(drug_norm, rna_norm.transpose(1, 2))

        elif self.similarity_method == 'mlp':
            drug_expanded = drug_feats.unsqueeze(2).expand(-1, -1, num_rna, -1)
            rna_expanded = rna_feats.unsqueeze(1).expand(-1, num_drug, -1, -1)
            paired_feats = torch.cat([drug_expanded, rna_expanded], dim=-1)
            similarity_matrix = self.similarity_mlp(paired_feats).squeeze(-1)

        elif self.similarity_method == 'bilinear':
            similarity_scores = []
            for i in range(num_drug):
                for j in range(num_rna):
                    sim = self.bilinear(drug_feats[:, i, :], rna_feats[:, j, :])
                    similarity_scores.append(sim)
            similarity_matrix = torch.stack(similarity_scores, dim=1).view(batch_size, num_drug, num_rna)

        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

        return similarity_matrix

    def aggregate_similarity_features(self, drug_feats, rna_feats, similarity_matrix):
        """聚合特征 (V-SIMPLE)，处理单个样本"""
        # 输入形状: drug_feats [N_drug, D], rna_feats [N_rna, D], sim_matrix [N_drug, N_rna]
        if self.aggregation_method == 'attention':
             # 注意：MultiheadAttention需要[SeqLen, Batch, Dim]，我们将Batch=1
            drug_feats_b1 = drug_feats.unsqueeze(1) # [N_drug, 1, D]
            rna_feats_b1 = rna_feats.unsqueeze(1)   # [N_rna, 1, D]

            drug_query = self.drug_query_proj(drug_feats.mean(dim=0, keepdim=True)).unsqueeze(0) # [1, 1, D]
            rna_query = self.rna_query_proj(rna_feats.mean(dim=0, keepdim=True)).unsqueeze(0)   # [1, 1, D]
            
            # self-attention
            aggregated_drug, _ = self.drug_attention(drug_query, drug_feats_b1, drug_feats_b1)
            aggregated_drug = aggregated_drug.squeeze(0).squeeze(0) # -> [D]

            aggregated_rna, _ = self.rna_attention(rna_query, rna_feats_b1, rna_feats_b1)
            aggregated_rna = aggregated_rna.squeeze(0).squeeze(0) # -> [D]

        elif self.aggregation_method in ['max', 'mean']:
            if similarity_matrix.numel() == 0:
                agg_sim_per_drug = torch.zeros(drug_feats.size(0), device=drug_feats.device)
                agg_sim_per_rna = torch.zeros(rna_feats.size(0), device=rna_feats.device)
            else:
                if self.aggregation_method == 'max':
                    agg_sim_per_drug, _ = similarity_matrix.max(dim=1) # [N_drug]
                    agg_sim_per_rna, _ = similarity_matrix.max(dim=0)  # [N_rna]
                else: # mean
                    agg_sim_per_drug = similarity_matrix.mean(dim=1)
                    agg_sim_per_rna = similarity_matrix.mean(dim=0)
            
            drug_weights = F.softmax(agg_sim_per_drug, dim=0).unsqueeze(1) # [N_drug, 1]
            rna_weights = F.softmax(agg_sim_per_rna, dim=0).unsqueeze(1)   # [N_rna, 1]
            
            aggregated_drug = (drug_feats * drug_weights).sum(dim=0)
            aggregated_rna = (rna_feats * rna_weights).sum(dim=0)
        else:
             raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        stats = self._compute_similarity_stats(similarity_matrix)
        
        return aggregated_drug, aggregated_rna, stats

    def forward(self, drug_substructures, rna_motifs, rna_motifs_batch):
        """
        前向传播 (V7 - 循环处理，逻辑最清晰)

        Args:
            drug_substructures (Tensor): [B, num_drug_sub, D_drug]
            rna_motifs (Tensor): [N_total_motifs, D_motif], RNA高层基序的扁平化表示
            rna_motifs_batch (Tensor): [N_total_motifs], 基序对应的batch索引
        """
        batch_size = drug_substructures.size(0)
        device = drug_substructures.device

        # --- 投影特征 ---
        proj_drug_sub = self.drug_proj(self.dropout(drug_substructures)) # [B, N_drug, D_proj]
        proj_rna_motifs = self.rna_proj(self.dropout(rna_motifs)) if rna_motifs.numel() > 0 else rna_motifs

        # --- 初始化用于收集结果的列表 ---
        all_ssim_features = []
        all_similarity_matrices = []

        # --- 逐个样本处理 ---
        for i in range(batch_size):
            # 1. 提取当前样本的数据
            current_drug_feats = proj_drug_sub[i] # [N_drug, D_proj]
            
            # 找到属于当前样本的RNA基元
            sample_mask = (rna_motifs_batch == i)
            if sample_mask.any():
                current_rna_feats = proj_rna_motifs[sample_mask] # [N_rna_i, D_proj]
            else:
                # 如果这个样本没有RNA基元，创建一个空的张量
                current_rna_feats = torch.empty(0, self.projection_dim, device=device)

            # 2. 计算单个样本的相似性矩阵
            if current_rna_feats.numel() > 0:
                sim_matrix_i = self.compute_pairwise_similarity(
                    current_drug_feats.unsqueeze(0), # -> [1, N_drug, D]
                    current_rna_feats.unsqueeze(0)   # -> [1, N_rna_i, D]
                ).squeeze(0) # -> [N_drug, N_rna_i]
            else:
                sim_matrix_i = torch.empty(current_drug_feats.size(0), 0, device=device)
            
            all_similarity_matrices.append(sim_matrix_i)

            # 3. 聚合特征
            agg_drug_i, agg_rna_i, stats_i = self.aggregate_similarity_features(
                current_drug_feats, current_rna_feats, sim_matrix_i
            )

            # 4. 拼接并投影得到最终的ssim特征
            final_features_i = torch.cat([agg_drug_i, agg_rna_i, stats_i], dim=0)
            ssim_features_i = self.final_proj(final_features_i)
            all_ssim_features.append(ssim_features_i)

        # --- 批处理结果 ---
        ssim_features_batch = torch.stack(all_ssim_features, dim=0)

        # 对于 similarity_matrix，由于大小不一，无法直接堆叠。
        # 我们可以返回一个列表，或者为辅助损失计算一个批处理的平均值。
        # 这里为了简单，我们只返回特征。如果辅助损失需要，可以进一步处理。
        # 为了兼容 compute_loss，我们可以返回一个填充后的矩阵。
        max_rna_len = max([m.size(1) for m in all_similarity_matrices] if all_similarity_matrices else [0])
        padded_sim_matrices = torch.zeros(batch_size, proj_drug_sub.size(1), max_rna_len, device=device)
        for i, m in enumerate(all_similarity_matrices):
            if m.numel() > 0:
                padded_sim_matrices[i, :, :m.size(1)] = m
        
        return ssim_features_batch, padded_sim_matrices