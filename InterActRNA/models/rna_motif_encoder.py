# models/rna_motif_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from data_processing.utils import parse_rna_motifs_simple
from utils.logger_utils import log_print


class RNAMotifEncoder(nn.Module):
    """
    一个全新的模块，用于学习RNA的基序级表示。
    它接收底层的核苷酸表示，并输出高层的基序表示。
    """

    def __init__(self, node_input_dim, motif_embedding_dim, num_heads=4, dropout=0.1, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.motif_embedding_dim = motif_embedding_dim

        if motif_embedding_dim % num_heads != 0:
            raise ValueError(
                f"motif_embedding_dim ({motif_embedding_dim}) must be divisible by num_heads ({num_heads})")

        # 使用GAT在动态构建的基序图上进行消息传递
        self.gat_conv1 = GATConv(node_input_dim, motif_embedding_dim // num_heads, heads=num_heads, dropout=dropout)
        self.gat_conv2 = GATConv(motif_embedding_dim, motif_embedding_dim, heads=1, concat=False, dropout=dropout)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, rna_node_features, rna_batch_idx, rna_dot_bracket_strings):
        """
        Args:
            rna_node_features (Tensor): [N_total_nodes, D_node], 底层核苷酸表示
            rna_batch_idx (Tensor): [N_total_nodes], 核苷酸对应的batch索引
            rna_dot_bracket_strings (list[str]): batch中每个RNA的二级结构字符串

        Returns:
            motif_features (Tensor): [N_total_motifs, D_motif], 所有基序的最终表示
            motif_batch_idx (Tensor): [N_total_motifs], 基序对应的batch索引
        """
        device = rna_node_features.device
        batch_size = rna_batch_idx.max().item() + 1 if rna_batch_idx.numel() > 0 else 0

        all_motifs_initial_features = []
        all_motif_adj_edges = []
        motif_batch_idx_list = []
        current_motif_idx_offset = 0

        for i in range(batch_size):
            node_mask = (rna_batch_idx == i)
            current_rna_nodes = rna_node_features[node_mask]

            if current_rna_nodes.size(0) == 0:
                continue

            db_string = rna_dot_bracket_strings[i]
            motifs = parse_rna_motifs_simple(db_string)

            initial_motif_feats_this_rna = []
            motif_map = {}

            if motifs['stem']:
                stem_indices = torch.tensor(motifs['stem'], dtype=torch.long, device=device)
                if len(stem_indices) > 0:
                    initial_motif_feats_this_rna.append(current_rna_nodes[stem_indices].mean(dim=0))
                    motif_map['stem'] = len(motif_map)

            if motifs['loop']:
                loop_indices = torch.tensor(motifs['loop'], dtype=torch.long, device=device)
                if len(loop_indices) > 0:
                    initial_motif_feats_this_rna.append(current_rna_nodes[loop_indices].mean(dim=0))
                    motif_map['loop'] = len(motif_map)

            num_motifs_in_this_rna = len(initial_motif_feats_this_rna)
            if num_motifs_in_this_rna == 0:
                continue

            all_motifs_initial_features.append(torch.stack(initial_motif_feats_this_rna))

            if 'stem' in motif_map and 'loop' in motif_map:
                stem_idx_local = motif_map['stem']
                loop_idx_local = motif_map['loop']
                all_motif_adj_edges.extend([
                    [stem_idx_local + current_motif_idx_offset, loop_idx_local + current_motif_idx_offset],
                    [loop_idx_local + current_motif_idx_offset, stem_idx_local + current_motif_idx_offset]
                ])

            motif_batch_idx_list.extend([i] * num_motifs_in_this_rna)
            current_motif_idx_offset += num_motifs_in_this_rna

        if not all_motifs_initial_features:
            return torch.empty(0, self.motif_embedding_dim, device=device), torch.empty(0, dtype=torch.long,
                                                                                        device=device)

        initial_motif_features_batch = torch.cat(all_motifs_initial_features, dim=0)
        motif_adj_batch = torch.tensor(all_motif_adj_edges, dtype=torch.long,
                                       device=device).t().contiguous() if all_motif_adj_edges else torch.empty((2, 0),
                                                                                                               dtype=torch.long,
                                                                                                               device=device)
        motif_batch_idx = torch.tensor(motif_batch_idx_list, dtype=torch.long, device=device)

        x = self.gat_conv1(initial_motif_features_batch, motif_adj_batch)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.gat_conv2(x, motif_adj_batch)

        return x, motif_batch_idx