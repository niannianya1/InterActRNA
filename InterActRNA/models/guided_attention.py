# models/guided_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        # 标准的多头注意力设置
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # 投影层
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        # 输出层
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_nodes, query_batch_idx, key_nodes, key_batch_idx):
        """
        用key_nodes的信息来更新query_nodes。
        """
        # 1. 投影
        q = self.q_proj(query_nodes)
        k = self.k_proj(key_nodes)
        v = self.v_proj(key_nodes)

        # 2. 生成注意力mask，确保注意力只在同一批次内的样本间计算
        # 注意：MHA的mask中，True代表“这个位置不被允许attend”
        attn_mask = (query_batch_idx.unsqueeze(1) != key_batch_idx.unsqueeze(0))

        # 3. 计算注意力
        # batch_first=True，输入形状为 [B, N, D]。我们将整个批次的节点看作一个大序列，B=1
        ctx, _ = self.mha(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=attn_mask)
        ctx = ctx.squeeze(0)

        # 4. 残差连接和层归一化
        updated_query_nodes = self.layer_norm(query_nodes + self.dropout(self.out_proj(ctx)))

        return updated_query_nodes