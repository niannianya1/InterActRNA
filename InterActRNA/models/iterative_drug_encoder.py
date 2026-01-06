# models/iterative_drug_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter

from utils.logger_utils import log_print


class IterativeDrugEncoder(nn.Module):
    """
    基于SA-DDI思想的迭代药物编码器
    通过多次消息传递学习不同半径的子结构表示
    支持返回所有迭代步骤的特征用于SSIM模块
    """

    def __init__(self,
                 node_emb_dim=64,
                 hidden_dim=128,
                 num_iterations=10,
                 output_dim=128,
                 atom_vocab_size=119,
                 chirality_vocab_size=4,
                 dropout_rate=0.3,
                 return_all_iterations=False,  # <<< 新增：是否返回所有迭代步骤的特征 >>>
                 debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim
        self.return_all_iterations = return_all_iterations  # <<< 新增 >>>

        # 原子嵌入层
        if chirality_vocab_size and chirality_vocab_size > 0:
            atom_emb_dim = node_emb_dim // 2
            chirality_emb_dim = node_emb_dim - atom_emb_dim
            self.atom_embedding = nn.Embedding(atom_vocab_size, atom_emb_dim)
            self.chirality_embedding = nn.Embedding(chirality_vocab_size, chirality_emb_dim)
        else:
            self.atom_embedding = nn.Embedding(atom_vocab_size, node_emb_dim)
            self.chirality_embedding = None

        # 初始特征投影
        self.initial_proj = nn.Linear(node_emb_dim, hidden_dim)

        # GIN卷积层（用于迭代）
        self.gin_convs = nn.ModuleList()
        for _ in range(num_iterations):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.gin_convs.append(GINConv(mlp, train_eps=True))

        # 子结构注意力参数
        self.substructure_attention = SubstructureAttention(hidden_dim, num_iterations)

        # 最终输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # <<< 新增：用于SSIM的子结构特征投影 >>>
        if self.return_all_iterations:
            self.substruct_proj = nn.Linear(hidden_dim, hidden_dim)  # 可以调整投影维度

    def forward(self, data_batch, h_initial=None): # 增加一个可选参数
        """
        Args:
            data_batch: PyG batch data
        Returns:
            如果return_all_iterations=False:
                graph_repr: 图级表示 [batch_size, output_dim]
                node_repr: 节点级表示 [total_nodes, output_dim]
            如果return_all_iterations=True:
                graph_repr: 图级表示 [batch_size, output_dim]
                node_repr: 节点级表示 [total_nodes, output_dim]
                all_substruct_reprs: 所有迭代步骤的子结构表示 [batch_size, num_iterations, hidden_dim]
        """
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch

        # <--- 核心修正：修复双重投影的逻辑 ---
        if h_initial is None:
            # 原子特征嵌入
            atom_features = self.atom_embedding(x[:, 0])
            if self.chirality_embedding is not None and x.size(1) > 1:
                chirality_features = self.chirality_embedding(x[:, 1])
                h_embedded = torch.cat([atom_features, chirality_features], dim=-1)
            else:
                h_embedded = atom_features
            # 初始投影
            h = self.initial_proj(h_embedded)
        else:
            # 如果提供了 h_initial，直接使用，不再投影
            h = h_initial
        # <--- 修正结束 ---

        # 存储每次迭代的表示（不同半径的子结构）
        substructure_reprs = []
        all_node_features = []  # <<< 新增：存储所有迭代步骤的节点特征 >>>
        
        if self.debug_mode:
            log_print(f"[IterativeDrugEncoder] Starting {self.num_iterations} GIN iterations...", level="debug")
        
        # 迭代学习不同半径的子结构
        for i, gin_conv in enumerate(self.gin_convs):
            h = gin_conv(h, edge_index)
            h = self.dropout(h)

            # 池化得到图级表示（当前半径的子结构表示）
            graph_repr_i = global_mean_pool(h, batch)
            substructure_reprs.append(graph_repr_i)
            
            # <<< 新增：如果需要返回所有迭代特征，则存储节点特征 >>>
            if self.return_all_iterations:
                all_node_features.append(h.clone())

            if self.debug_mode:
                print(f"  Iteration {i + 1}: node_features_mean={h.mean().item():.4f}, "
                      f"graph_repr_mean={graph_repr_i.mean().item():.4f}")

        # 堆叠所有半径的子结构表示
        all_substructures = torch.stack(substructure_reprs, dim=-1)  # [batch_size, hidden_dim, num_iterations]
        
        if self.debug_mode:
            log_print("[IterativeDrugEncoder] Calling SubstructureAttention...", level="debug")
        
        # 子结构注意力融合
        final_graph_repr = self.substructure_attention(all_substructures)
        
        if self.debug_mode:
            # 我们可以打印一下注意力权重，看看它学到了什么
            if hasattr(self.substructure_attention, 'latest_weights'):
                weights_str = ", ".join(
                    [f"{w:.3f}" for w in self.substructure_attention.latest_weights[0, 0, :].tolist()])
                log_print(f"  [SubstructureAttention] Weights: [{weights_str}]", level="debug")
            log_print(f"  [SubstructureAttention] Fused Graph Repr mean: {final_graph_repr.mean().item():.4f}",
                      level="debug")

        # 最终投影
        final_graph_repr = self.output_proj(final_graph_repr)
        final_node_repr = self.output_proj(h)  # 使用最后一次迭代的节点特征

        if self.debug_mode:
            print(f"  Final graph repr shape: {final_graph_repr.shape}")
            print(f"  Final node repr shape: {final_node_repr.shape}")

        # <<< 新增：处理子结构特征用于SSIM >>>
        if self.return_all_iterations:
            # 为每个迭代步骤创建子结构表示
            substruct_features = []
            for iter_nodes in all_node_features:
                # 对每个迭代步骤的节点特征进行池化
                iter_substruct = global_mean_pool(self.substruct_proj(iter_nodes), batch)
                substruct_features.append(iter_substruct)
            
            # 堆叠成 [batch_size, num_iterations, hidden_dim]
            all_substruct_reprs = torch.stack(substruct_features, dim=1)
            
            if self.debug_mode:
                print(f"  All substructure reprs shape: {all_substruct_reprs.shape}")
            
            return final_graph_repr, final_node_repr, all_substruct_reprs
        else:
            return final_graph_repr, final_node_repr

    def get_substructure_attention_weights(self):
        """获取子结构注意力权重，用于可视化分析"""
        if hasattr(self.substructure_attention, 'latest_weights'):
            return self.substructure_attention.latest_weights.detach()
        return None

    def get_all_iterations_features(self, data_batch):
        """
        专门用于获取所有迭代步骤特征的方法
        主要用于SSIM模块或其他分析用途
        """
        # 临时设置返回所有迭代特征
        original_return_setting = self.return_all_iterations
        self.return_all_iterations = True
        
        try:
            results = self.forward(data_batch)
            if len(results) == 3:
                _, _, all_substruct_reprs = results
                return all_substruct_reprs
            else:
                return None
        finally:
            # 恢复原始设置
            self.return_all_iterations = original_return_setting


class SubstructureAttention(nn.Module):
    """
    子结构注意力模块，用于融合不同半径的子结构表示
    类似SA-DDI中的substructure attention
    """

    def __init__(self, hidden_dim, num_iterations):
        super().__init__()
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim

        # 注意力参数
        self.attention_weights = nn.Parameter(torch.zeros(1, hidden_dim, num_iterations))
        self.attention_bias = nn.Parameter(torch.zeros(1, 1, num_iterations))

        # 初始化
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, substructure_features):
        """
        Args:
            substructure_features: [batch_size, hidden_dim, num_iterations]
        Returns:
            weighted_repr: [batch_size, hidden_dim]
        """
        # 计算注意力分数
        scores = (substructure_features * self.attention_weights).sum(dim=1, keepdim=True) + self.attention_bias
        # [batch_size, 1, num_iterations]

        # Softmax归一化
        attention_weights = torch.softmax(scores, dim=-1)
        self.latest_weights = attention_weights.detach()  # 保存权重用于调试

        # 加权求和
        weighted_repr = (substructure_features * attention_weights).sum(dim=-1)

        return weighted_repr

class TargetAwareFusion(nn.Module):
        """
        一个基于交叉注意力的融合模块。
        它使用RNA的表示作为Query，来动态地融合药物的多尺度子结构。
        """

        def __init__(self, drug_dim, rna_dim, num_heads=4, dropout=0.1):
            super().__init__()
            if drug_dim % num_heads != 0:
                raise ValueError(f"drug_dim ({drug_dim}) must be divisible by num_heads ({num_heads})")

            self.query_proj = nn.Linear(rna_dim, drug_dim)

            self.cross_attention = nn.MultiheadAttention(
                embed_dim=drug_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            self.layer_norm = nn.LayerNorm(drug_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, drug_substructures, rna_context):
            """
            Args:
                drug_substructures (Tensor): [batch_size, num_iterations, drug_dim]
                rna_context (Tensor): [batch_size, rna_dim]
            Returns:
                Tensor: [batch_size, drug_dim]
            """
            query = self.query_proj(rna_context).unsqueeze(1)

            attn_output, self.latest_weights = self.cross_attention(
                query=query,
                key=drug_substructures,
                value=drug_substructures
            )

            fused_repr = attn_output.squeeze(1)

            # 使用残差连接和层归一化
            residual = drug_substructures.mean(dim=1)
            fused_repr = self.layer_norm(residual + self.dropout(fused_repr))

            return fused_repr