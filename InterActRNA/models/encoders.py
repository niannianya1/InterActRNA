# drug_rna_mili_pyg/models/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, GCNConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


# GINMLP (保持不变)
class GINMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, use_bn=True, use_relu=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_bn = use_bn
        self.use_relu = use_relu
        if use_bn: self.bns = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.use_bn and hasattr(self, 'bns') and i < len(self.bns): x = self.bns[i](x)
                if self.use_relu: x = F.relu(x)
        return x


# GINEncoder (修改了返回值)
class GINEncoder(nn.Module):
    def __init__(self, node_emb_dim, hidden_channels_mlp, num_gin_layers,
                 output_graph_emb_dim, atom_vocab_size, chirality_vocab_size,
                 dropout_rate=0.5, jk_mode='last', readout='mean', debug_mode=False): # 添加 debug_mode 参数
        super().__init__()
        self.debug_mode = debug_mode
        self.num_gin_layers = num_gin_layers
        self.jk_mode = jk_mode.lower()
        self.dropout = nn.Dropout(dropout_rate)

        if not atom_vocab_size: raise ValueError("atom_vocab_size must be provided for GINEncoder.")
        atom_emb_output_dim = node_emb_dim
        chirality_emb_output_dim = 0
        if chirality_vocab_size and chirality_vocab_size > 0:
            atom_emb_output_dim = node_emb_dim // 2
            chirality_emb_output_dim = node_emb_dim - atom_emb_output_dim
            self.chirality_embedding = nn.Embedding(chirality_vocab_size, chirality_emb_output_dim)
        else:
            self.chirality_embedding = None
        self.atom_type_embedding = nn.Embedding(atom_vocab_size, atom_emb_output_dim)

        gin_input_node_dim = node_emb_dim

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gin_layers):
            mlp = GINMLP(gin_input_node_dim, hidden_channels_mlp, gin_input_node_dim, num_layers=2, use_bn=True)
            self.gnn_layers.append(GINConv(nn=mlp, train_eps=True))

        if self.jk_mode not in ['last', 'none', None, '']:
            self.jk_layer = JumpingKnowledge(mode=self.jk_mode, channels=gin_input_node_dim, num_layers=num_gin_layers)
        else:
            self.jk_layer = None

        if readout == 'sum':
            self.readout_pool = global_add_pool
        elif readout == 'mean':
            self.readout_pool = global_mean_pool
        elif readout == 'max':
            self.readout_pool = global_max_pool
        else:
            raise ValueError(f"Unsupported readout: {readout}")

        if self.jk_layer is None or self.jk_mode in ['last', 'none', '', None]:
            self.jk_output_dim = gin_input_node_dim
        elif self.jk_mode == 'cat':
            self.jk_output_dim = num_gin_layers * gin_input_node_dim
        else:
            self.jk_output_dim = gin_input_node_dim

        self.final_fc = nn.Linear(self.jk_output_dim,
                                  output_graph_emb_dim) if self.jk_output_dim != output_graph_emb_dim else nn.Identity()

        # <<< 新增：节点特征到最终输出维度的投影层 >>>
        # 用于cross-attention之前统一维度
        self.node_feature_out_dim = output_graph_emb_dim
        self.node_proj = nn.Linear(self.jk_output_dim,
                                   self.node_feature_out_dim) if self.jk_output_dim != self.node_feature_out_dim else nn.Identity()

    def forward(self, data_batch):
        x, edge_index, batch_idx = data_batch.x, data_batch.edge_index, data_batch.batch

        atom_type_x_embedded = self.atom_type_embedding(x[:, 0])
        chirality_x_embedded = None
        if self.chirality_embedding is not None and x.size(1) > 1:
            chirality_x_embedded = self.chirality_embedding(x[:, 1])
            h = torch.cat([atom_type_x_embedded, chirality_x_embedded], dim=-1)
        else:
            h = atom_type_x_embedded

        all_layer_node_feats = []
        for i, gin_layer in enumerate(self.gnn_layers):
            h = gin_layer(h, edge_index)
            h = F.relu(h)
            if i < self.num_gin_layers - 1:
                h = self.dropout(h)
            all_layer_node_feats.append(h)

        if self.jk_layer is not None:
            final_node_feats = self.jk_layer(all_layer_node_feats)
        else:
            final_node_feats = all_layer_node_feats[-1]

        final_node_feats_dropped = self.dropout(final_node_feats)

        graph_feats_before_fc = self.readout_pool(final_node_feats_dropped, batch_idx)

        graph_feats = self.final_fc(graph_feats_before_fc)

        # <<< 新增：计算并返回节点级特征 >>>
        node_feats = self.node_proj(final_node_feats_dropped)
        if self.debug_mode:  # 假设 GINEncoder 也接收 debug_mode 参数
            print(f"  GINEncoder Output Shapes: graph_feats={graph_feats.shape}, node_feats={node_feats.shape}")
            print(
                f"  GINEncoder Node Feats Stats: mean={node_feats.mean().item():.4f}, std={node_feats.std().item():.4f}, min={node_feats.min().item():.4f}, max={node_feats.max().item():.4f}")
        return graph_feats, node_feats


# SingleViewRNAGNN (修改了返回值)
class SingleViewRNAGNN(nn.Module):
    def __init__(self, input_embedding_dim, hidden_dim, output_node_dim,
                 num_layers=2, gnn_type='gat', gat_heads=4, dropout=0.2, readout_pool_type='mean',
                 predict_node_scores=False, debug_mode=False): # 添加 debug_mode 参数
        super().__init__()
        self.debug_mode = debug_mode # 存储 debug_mode
        self.convs = nn.ModuleList();
        current_dim = input_embedding_dim
        self.gnn_type = gnn_type.lower();
        self.predict_node_scores = predict_node_scores
        self.num_layers = num_layers
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            if self.gnn_type == 'gcn':
                out_c = output_node_dim if is_last_layer else hidden_dim
                self.convs.append(GCNConv(current_dim, out_c));
                current_dim = out_c
            elif self.gnn_type == 'gat':
                if is_last_layer:
                    out_channels_this_layer = output_node_dim;
                    final_gat_heads = 1;
                    concat_heads = False
                else:
                    if hidden_dim % gat_heads != 0: raise ValueError(
                        f"GAT hidden_dim ({hidden_dim}) must be divisible by gat_heads ({gat_heads}) for intermediate layers.")
                    out_channels_this_layer = hidden_dim // gat_heads;
                    final_gat_heads = gat_heads;
                    concat_heads = True
                self.convs.append(
                    GATConv(current_dim, out_channels_this_layer, heads=final_gat_heads, concat=concat_heads,
                            dropout=dropout))
                current_dim = out_channels_this_layer * final_gat_heads if concat_heads else out_channels_this_layer
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
        if current_dim != output_node_dim and num_layers > 0:
            self.final_node_transform = nn.Linear(current_dim, output_node_dim)
        else:
            self.final_node_transform = nn.Identity()
        self.activation = nn.ELU();
        self.dropout_gnn = nn.Dropout(dropout)

        if readout_pool_type == 'sum':
            self.readout_pool_fn = global_add_pool
        elif readout_pool_type == 'mean':
            self.readout_pool_fn = global_mean_pool
        elif readout_pool_type == 'max':
            self.readout_pool_fn = global_max_pool
        else:
            raise ValueError(f"Unsupported readout_pool_type: {readout_pool_type}")

        if self.predict_node_scores: self.node_score_predictor = nn.Linear(output_node_dim, 1)

    def forward(self, embedded_x, edge_index, batch_idx):
        h = embedded_x
        for i, conv_layer in enumerate(self.convs):
            h = conv_layer(h, edge_index);
            h = self.activation(h)
            if i < self.num_layers - 1: h = self.dropout_gnn(h)
        h_nodes_final = self.final_node_transform(h)

        graph_embedding = self.readout_pool_fn(h_nodes_final, batch_idx)

        if self.predict_node_scores:
            if not hasattr(self, 'node_score_predictor'): raise RuntimeError(
                "predict_node_scores is True, but node_score_predictor is not initialized.")
            node_scores = self.node_score_predictor(h_nodes_final).squeeze(-1)
            # <<< 返回节点特征 >>>
            return graph_embedding, h_nodes_final, node_scores
        else:
            # <<< 返回节点特征 >>>
            return graph_embedding, h_nodes_final


# MultiViewRNAEncoder (修改了返回值和内部逻辑)
class MultiViewRNAEncoder(nn.Module):
    def __init__(self, input_nucleotide_vocab_size, nucleotide_embedding_dim,
                 view_seq_gnn_config, view_pair_gnn_config,
                 fusion_method='concat', fused_representation_dim=None, dropout_after_fusion=0.0,
                 predict_node_scores_for_subgraph=False, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.nucleotide_embedding = nn.Embedding(input_nucleotide_vocab_size, nucleotide_embedding_dim)
        self.predict_node_scores_for_subgraph = predict_node_scores_for_subgraph

        # 序列
        view_seq_gnn_config_updated = view_seq_gnn_config.copy()
        view_seq_gnn_config_updated['predict_node_scores'] = self.predict_node_scores_for_subgraph
        self.gnn_seq_view = SingleViewRNAGNN(input_embedding_dim=nucleotide_embedding_dim, **view_seq_gnn_config_updated)
        # 结构
        view_pair_gnn_config_updated = view_pair_gnn_config.copy()
        view_pair_gnn_config_updated['predict_node_scores'] = self.predict_node_scores_for_subgraph
        self.gnn_pair_view = SingleViewRNAGNN(input_embedding_dim=nucleotide_embedding_dim, **view_pair_gnn_config_updated)

        self.fusion_method = fusion_method.lower()

        dim_seq = view_seq_gnn_config['output_node_dim']
        dim_pair = view_pair_gnn_config['output_node_dim']

        if self.fusion_method == 'concat':
            current_fused_dim = dim_seq + dim_pair
        elif self.fusion_method in ['mean', 'sum']:
            if dim_seq != dim_pair:
                raise ValueError("For 'mean' or 'sum' fusion, output_node_dim of GNNs must be same.")
            current_fused_dim = dim_seq
        else:
            raise ValueError(f"Unsupported fusion_method for graph embeddings: {self.fusion_method}")

        if fused_representation_dim is not None and current_fused_dim != fused_representation_dim:
            self.fusion_mlp_graph = nn.Linear(current_fused_dim, fused_representation_dim)
            self.final_output_dim = fused_representation_dim
        else:
            self.fusion_mlp_graph = nn.Identity()
            self.final_output_dim = current_fused_dim

        self.node_feature_out_dim = self.final_output_dim
        self.fusion_mlp_node = nn.Linear(current_fused_dim, self.node_feature_out_dim) if current_fused_dim != self.node_feature_out_dim else nn.Identity()

        self.dropout_final = nn.Dropout(dropout_after_fusion) if dropout_after_fusion > 0 else nn.Identity()
        if self.predict_node_scores_for_subgraph:
            self.alpha_param_for_node_scores = nn.Parameter(torch.tensor(0.0))

    # --- vvvv  vvvv ---
    def forward(self, rna_data_batch, return_embedding=False, initial_embedding=None):
        # 1. 获取初始节点嵌入
        if initial_embedding is None:
            x_indices = rna_data_batch.x.squeeze(-1) if rna_data_batch.x.dim() > 1 else rna_data_batch.x
            embedded_x = self.nucleotide_embedding(x_indices)
        else:
            # 如果提供了外部嵌入，直接使用
            embedded_x = initial_embedding

        if return_embedding:
            # 如果只是想获取初始嵌入，提前返回
            return None, embedded_x, None

        total_node_scores = None
        if self.predict_node_scores_for_subgraph:
            h_graph_seq, h_nodes_seq, scores_seq = self.gnn_seq_view(embedded_x, rna_data_batch.edge_index_seq, rna_data_batch.batch)
            h_graph_pair, h_nodes_pair, scores_pair = self.gnn_pair_view(embedded_x, rna_data_batch.edge_index_pair, rna_data_batch.batch)
            alpha = torch.sigmoid(self.alpha_param_for_node_scores)
            total_node_scores = alpha * scores_seq + (1 - alpha) * scores_pair
        else:
            # 只在序列边上工作
            h_graph_seq, h_nodes_seq = self.gnn_seq_view(embedded_x, rna_data_batch.edge_index_seq, rna_data_batch.batch)
            # 只在配对边上工作
            h_graph_pair, h_nodes_pair = self.gnn_pair_view(embedded_x, rna_data_batch.edge_index_pair, rna_data_batch.batch)
        # 3. 信息汇总 拼接
        if self.fusion_method == 'concat':
            fused_graph_repr = torch.cat((h_graph_seq, h_graph_pair), dim=1)
            fused_node_repr = torch.cat((h_nodes_seq, h_nodes_pair), dim=1)
        elif self.fusion_method == 'mean':
            fused_graph_repr = (h_graph_seq + h_graph_pair) / 2
            fused_node_repr = (h_nodes_seq + h_nodes_pair) / 2
        elif self.fusion_method == 'sum':
            fused_graph_repr = h_graph_seq + h_graph_pair
            fused_node_repr = h_nodes_seq + h_nodes_pair
        
        final_fused_graph_representation = self.dropout_final(self.fusion_mlp_graph(fused_graph_repr))
        final_fused_node_representation = self.dropout_final(self.fusion_mlp_node(fused_node_repr))
        
        if self.debug_mode:
            print(f"MultiViewRNAEncoder Node Feats Stats: mean={final_fused_node_representation.mean().item():.4f}")
            
        return final_fused_graph_representation, final_fused_node_representation, total_node_scores

# SubgraphCrossViewAttention
class SubgraphCrossViewAttention(nn.Module):
    def __init__(self, seq_view_dim, pair_view_dim, attention_hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.attention_hidden_dim = attention_hidden_dim
        self.WQ_s_on_p = nn.Linear(seq_view_dim, attention_hidden_dim, bias=False)
        self.WK_p_for_s = nn.Linear(pair_view_dim, attention_hidden_dim, bias=False)
        self.WV_p = nn.Linear(pair_view_dim, output_dim, bias=False)

        self.WQ_p_on_s = nn.Linear(pair_view_dim, attention_hidden_dim, bias=False)
        self.WK_s_for_p = nn.Linear(seq_view_dim, attention_hidden_dim, bias=False)
        self.WV_s = nn.Linear(seq_view_dim, output_dim, bias=False)

        self.fc_out = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, h_seq_sub_batch, h_pair_sub_batch):
        if h_seq_sub_batch is None and h_pair_sub_batch is None: return None
        if h_seq_sub_batch is None:
            return self.layer_norm(F.relu(self.WV_p(h_pair_sub_batch)))

        if h_pair_sub_batch is None:
            return self.layer_norm(F.relu(self.WV_s(h_seq_sub_batch)))

        Q_s = self.WQ_s_on_p(h_seq_sub_batch)
        K_p_for_s = self.WK_p_for_s(h_pair_sub_batch)
        V_p = self.WV_p(h_pair_sub_batch)
        attn_gate_s_on_p = torch.sigmoid(
            torch.sum(Q_s * K_p_for_s, dim=-1, keepdim=True) / (self.attention_hidden_dim ** 0.5))
        attended_V_p = attn_gate_s_on_p * V_p

        Q_p = self.WQ_p_on_s(h_pair_sub_batch)
        K_s_for_p = self.WK_s_for_p(h_seq_sub_batch)
        V_s = self.WV_s(h_seq_sub_batch)
        attn_gate_p_on_s = torch.sigmoid(
            torch.sum(Q_p * K_s_for_p, dim=-1, keepdim=True) / (self.attention_hidden_dim ** 0.5))
        attended_V_s = attn_gate_p_on_s * V_s

        fused_representation = attended_V_s + attended_V_p

        fused_representation = self.dropout(F.relu(self.fc_out(fused_representation)))
        fused_representation = self.layer_norm(fused_representation)

        return fused_representation