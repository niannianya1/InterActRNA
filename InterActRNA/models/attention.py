# drug_rna_mili_pyg/models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax  # PyG的scatter_softmax用于组内softmax

from utils.logger_utils import log_print


class PyGAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attention_hidden_dim):
        """
        注意力模块，用于计算药物整体表示与片段表示之间的注意力。

        Args:
            query_dim (int): 查询向量的维度 (例如, 整体药物嵌入的维度).
            key_dim (int): 键向量的维度 (例如, 药物片段嵌入的维度).
            attention_hidden_dim (int): 注意力机制内部投影后的维度.
        """
        super().__init__()
        self.attention_hidden_dim = attention_hidden_dim

        # 线性投影层
        self.WQ = nn.Linear(query_dim, attention_hidden_dim, bias=False)  # MILI的Attention没有偏置
        self.WK = nn.Linear(key_dim, attention_hidden_dim, bias=False)  # MILI的Attention没有偏置

        # MILI的Attention类中似乎没有Value的投影WV，它是直接用原始Key(substructure_feat)和softmax(att)相乘
        # 所以我们这里也遵循这个模式，只投影Q和K用于计算分数。

    def forward(self, query_batch, key_batch, original_mol_idx_for_keys):
        """
        计算注意力权重。z

        Args:
            query_batch (torch.Tensor): 批处理后的查询张量。
                形状: [num_original_mols, query_dim] (例如, h_mol_batch from global drug encoder)
            key_batch (torch.Tensor): 批处理后的键张量。
                形状: [total_num_keys_in_batch, key_dim] (例如, h_fragments_batch from fragment drug encoder)
            original_mol_idx_for_keys (torch.Tensor): 映射张量。
                形状: [total_num_keys_in_batch]，每个元素表示该键属于批次中哪个原始查询(分子)。

        Returns:
            torch.Tensor: 计算得到的注意力权重（已经过softmax）。
                形状: [total_num_keys_in_batch]，表示每个键相对于其对应查询的权重。
        """

        # 1. 投影查询和键
        # query_batch: [B, Dq], key_batch: [N_total_frags, Dk]
        # original_mol_idx_for_keys: [N_total_frags]

        Q_proj = self.WQ(query_batch)  # [B, Dh]
        K_proj = self.WK(key_batch)  # [N_total_frags, Dh]

        # 2. 将查询扩展以匹配键，用于计算点积注意力分数
        # 对于每个fragment (key)，我们需要其对应原始分子 (query) 的投影表示
        Q_expanded = Q_proj[original_mol_idx_for_keys]  # [N_total_frags, Dh]

        # 3. 计算注意力分数 (scaled dot-product)
        # Q_expanded 和 K_proj 都是 [N_total_frags, Dh]
        # 我们需要它们逐元素相乘后按隐藏维度求和
        attention_scores_flat = torch.sum(Q_expanded * K_proj, dim=1) / (self.attention_hidden_dim ** 0.5)
        # attention_scores_flat: [N_total_frags] (每个片段一个分数)

        # 4. 对每个原始分子的片段组应用Softmax
        # scatter_softmax会根据original_mol_idx_for_keys将attention_scores_flat分组，
        # 然后在每组内部独立计算softmax。
        attention_weights_flat = scatter_softmax(src=attention_scores_flat, index=original_mol_idx_for_keys, dim=0)
        # attention_weights_flat: [N_total_frags]

        return attention_weights_flat, attention_scores_flat  # 返回softmax后的权重和原始分数（用于计算反向注意力）


class DrugRNACrossAttention(nn.Module):
    def __init__(self, drug_dim, rna_dim, hidden_dim, num_heads, dropout=0.1, debug_mode=False): # 添加 debug_mode 参数
        """
        PyG兼容的药物-RNA节点级Cross-Attention模块。

        Args:
            drug_dim (int): 药物原子节点特征的维度。
            rna_dim (int): RNA核苷酸节点特征的维度。
            hidden_dim (int): 注意力机制内部的隐藏维度。必须能被num_heads整除。
            num_heads (int): 注意力头的数量。
            dropout (float): Dropout概率。
        """
        super().__init__()
        self.debug_mode = debug_mode
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 药物查询RNA
        self.to_q_drug = nn.Linear(drug_dim, hidden_dim, bias=False)
        self.to_k_rna = nn.Linear(rna_dim, hidden_dim, bias=False)
        self.to_v_rna = nn.Linear(rna_dim, hidden_dim, bias=False)
        self.out_drug = nn.Linear(hidden_dim, drug_dim)

        # RNA查询药物
        self.to_q_rna = nn.Linear(rna_dim, hidden_dim, bias=False)
        self.to_k_drug = nn.Linear(drug_dim, hidden_dim, bias=False)
        self.to_v_drug = nn.Linear(drug_dim, hidden_dim, bias=False)
        self.out_rna = nn.Linear(hidden_dim, rna_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_nodes, drug_batch_idx, rna_nodes, rna_batch_idx):
        """
        Args:
            drug_nodes (Tensor): 扁平化的药物原子节点特征 [total_drug_atoms, drug_dim]
            drug_batch_idx (Tensor): 药物原子对应的图索引 [total_drug_atoms]
            rna_nodes (Tensor): 扁平化的RNA核苷酸节点特征 [total_rna_nodes, rna_dim]
            rna_batch_idx (Tensor): RNA核苷酸对应的图索引 [total_rna_nodes]
        """

        # 添加调试信息
        if self.debug_mode:
            print(f"  [CrossAttention Debug]")
            print(f"    drug_batch_idx range: {drug_batch_idx.min().item()} - {drug_batch_idx.max().item()}")
            print(f"    rna_batch_idx range: {rna_batch_idx.min().item()} - {rna_batch_idx.max().item()}")
            print(f"    drug_batch_idx unique values: {drug_batch_idx.unique().tolist()}")
            print(f"    rna_batch_idx unique values: {rna_batch_idx.unique().tolist()}")

            # 检查是否有相同的batch索引
            common_indices = set(drug_batch_idx.unique().tolist()) & set(rna_batch_idx.unique().tolist())
            print(f"    Common batch indices: {sorted(common_indices)}")
            print(f"    Number of common indices: {len(common_indices)}")

        # 确保只在实际存在的共同batch索引之间计算注意力
        drug_batch_unique = drug_batch_idx.unique()
        rna_batch_unique = rna_batch_idx.unique()
        common_batch_indices = torch.tensor(
            sorted(set(drug_batch_unique.tolist()) & set(rna_batch_unique.tolist())),
            device=drug_batch_idx.device
        )

        if len(common_batch_indices) == 0:
            if self.debug_mode:
                print(f"    WARNING: No common batch indices found! Returning zero tensors.")
            # 返回零张量
            ctx_drug_nodes = torch.zeros_like(drug_nodes)
            ctx_rna_nodes = torch.zeros_like(rna_nodes)
            return self.dropout(ctx_drug_nodes), self.dropout(ctx_rna_nodes)

        # 只保留有共同batch索引的节点
        drug_mask = torch.isin(drug_batch_idx, common_batch_indices)
        rna_mask = torch.isin(rna_batch_idx, common_batch_indices)

        if self.debug_mode:
            print(f"    Drug nodes before filtering: {drug_nodes.shape[0]}, after: {drug_mask.sum().item()}")
            print(f"    RNA nodes before filtering: {rna_nodes.shape[0]}, after: {rna_mask.sum().item()}")

        # 如果过滤后节点太少，返回原始节点
        if drug_mask.sum() == 0 or rna_mask.sum() == 0:
            if self.debug_mode:
                print(f"    WARNING: After filtering, no valid nodes remain!")
            ctx_drug_nodes = torch.zeros_like(drug_nodes)
            ctx_rna_nodes = torch.zeros_like(rna_nodes)
            return self.dropout(ctx_drug_nodes), self.dropout(ctx_rna_nodes)

        # 创建更精确的mask
        # 重新映射batch索引到连续的0,1,2...
        batch_idx_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(common_batch_indices)}

        drug_batch_remapped = torch.tensor(
            [batch_idx_mapping[idx.item()] for idx in drug_batch_idx if idx.item() in batch_idx_mapping],
            device=drug_batch_idx.device)
        rna_batch_remapped = torch.tensor(
            [batch_idx_mapping[idx.item()] for idx in rna_batch_idx if idx.item() in batch_idx_mapping],
            device=rna_batch_idx.device)

        # 过滤节点
        drug_nodes_filtered = drug_nodes[drug_mask]
        rna_nodes_filtered = rna_nodes[rna_mask]

        # --- 1. 药物查询RNA ---
        q_drug = self.to_q_drug(drug_nodes_filtered)
        k_rna = self.to_k_rna(rna_nodes_filtered)
        v_rna = self.to_v_rna(rna_nodes_filtered)

        # 变成多头
        q_drug = q_drug.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k_rna = k_rna.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v_rna = v_rna.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        # 计算注意力分数
        scores_d_on_r = torch.matmul(q_drug, k_rna.transpose(-1, -2)) * self.scale

        # 创建更精确的注意力掩码
        mask = drug_batch_remapped.unsqueeze(1) == rna_batch_remapped.unsqueeze(0)

        if self.debug_mode:
            num_true_mask = mask.sum().item()
            total_mask_elements = mask.numel()
            log_print(
                f"  DEBUG (Filtered): Mask stats: num_true={num_true_mask}, total={total_mask_elements}, ratio_true={num_true_mask / total_mask_elements:.4f}",
                level="debug")

        # 应用掩码
        scores_d_on_r = scores_d_on_r.masked_fill(mask.unsqueeze(0) == 0, -1e9)
        attn_d_on_r = torch.softmax(scores_d_on_r, dim=-1)
        attn_d_on_r = self.dropout(attn_d_on_r)

        # 计算上下文
        ctx_from_rna = torch.matmul(attn_d_on_r, v_rna)
        ctx_from_rna = ctx_from_rna.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_dim)
        ctx_drug_filtered = self.out_drug(ctx_from_rna)

        # --- 2. RNA查询药物 (类似处理) ---
        q_rna = self.to_q_rna(rna_nodes_filtered)
        k_drug = self.to_k_drug(drug_nodes_filtered)
        v_drug = self.to_v_drug(drug_nodes_filtered)

        q_rna = q_rna.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k_drug = k_drug.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v_drug = v_drug.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        scores_r_on_d = torch.matmul(q_rna, k_drug.transpose(-1, -2)) * self.scale
        scores_r_on_d = scores_r_on_d.masked_fill(mask.t().unsqueeze(0) == 0, -1e9)
        attn_r_on_d = torch.softmax(scores_r_on_d, dim=-1)
        attn_r_on_d = self.dropout(attn_r_on_d)

        ctx_from_drug = torch.matmul(attn_r_on_d, v_drug)
        ctx_from_drug = ctx_from_drug.transpose(0, 1).contiguous().view(-1, self.num_heads * self.head_dim)
        ctx_rna_filtered = self.out_rna(ctx_from_drug)

        # 将结果映射回原始尺寸
        ctx_drug_nodes = torch.zeros_like(drug_nodes)
        ctx_rna_nodes = torch.zeros_like(rna_nodes)

        ctx_drug_nodes[drug_mask] = ctx_drug_filtered
        ctx_rna_nodes[rna_mask] = ctx_rna_filtered

        if self.debug_mode:
            print(f"  Cross-Attention Final Stats:")
            print(f"    ctx_drug_nodes non-zero ratio: {(ctx_drug_nodes != 0).float().mean().item():.4f}")
            print(f"    ctx_rna_nodes non-zero ratio: {(ctx_rna_nodes != 0).float().mean().item():.4f}")

        return self.dropout(ctx_drug_nodes), self.dropout(ctx_rna_nodes)
    
class FusionAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, feature_tensors_list):
        """
        Args:
            feature_tensors_list (list of Tensors): 
                一个包含多个特征张量的列表，例如 [drug_repr, rna_repr_high, rna_repr_low, ssim_repr]。
                每个张量的形状都是 [batch_size, feature_dim_i]。
        Returns:
            Tensor: 融合后的特征张量，形状为 [batch_size, total_feature_dim]。
        """
        # 1. 拼接所有特征
        combined_features = torch.cat(feature_tensors_list, dim=1) # [B, D_total]
        
        # 2. 计算每个特征维度的注意力分数
        attention_scores = self.attention_net(combined_features) # [B, 1]
        
        # 3. 创建注意力权重 (这里我们让整个特征向量共享一个权重)
        # 我们可以用 sigmoid 让权重在0-1之间，或者直接用原始分数
        attention_weights = torch.sigmoid(attention_scores) # [B, 1]

        # 4. 将注意力权重应用到原始的拼接特征上
        # 这相当于给每个样本的整个特征向量一个“重要性”加权
        weighted_features = combined_features * attention_weights
        
        # 5. (可选但推荐) 加入残差连接，保留原始信息
        fused_features = combined_features + weighted_features

        return fused_features