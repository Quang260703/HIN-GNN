import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn import build_encoder


class LinkScorer(nn.Module):
    """Bilinear / MLP scorer over two node embeddings."""

    def __init__(self, emb_dim, mode="mlp", hidden_dim=64):
        super().__init__()
        self.mode = mode
        if mode == "bilinear":
            self.W = nn.Parameter(torch.empty(emb_dim, emb_dim))
            nn.init.xavier_uniform_(self.W)
        elif mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown scorer mode: {mode}")

    def forward(self, emb_src, emb_dst):
        if self.mode == "bilinear":
            return torch.sum((emb_src @ self.W) * emb_dst, dim=-1)
        return self.mlp(torch.cat([emb_src, emb_dst], dim=-1)).squeeze(-1)


class ExpertRetrievalModel(nn.Module):
    """
    Full pipeline: heterogeneous encoder -> node embeddings ->
    link scorer between 'author' nodes and 'venue' (or 'paper') nodes.
    """

    def __init__(self, encoder_name, metadata, src_type="author",
                 dst_type="venue", emb_dim=64, scorer_mode="mlp",
                 encoder_kwargs=None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = build_encoder(
            encoder_name, metadata=metadata,
            out_channels=emb_dim, **encoder_kwargs
        )
        self.src_type = src_type
        self.dst_type = dst_type
        self.scorer = LinkScorer(emb_dim, mode=scorer_mode)

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def forward(self, x_dict, edge_index_dict, src_index, dst_index):
        """
        src_index, dst_index: LongTensors of node indices (within their
        type) for the candidate (author, venue/paper) pairs to score.
        """
        emb_dict = self.encode(x_dict, edge_index_dict)
        emb_src = emb_dict[self.src_type][src_index]
        emb_dst = emb_dict[self.dst_type][dst_index]
        return self.scorer(emb_src, emb_dst)

    @torch.no_grad()
    def rank_experts(self, x_dict, edge_index_dict, dst_index,
                      candidate_src_index):
        """Score all candidate authors against one target venue/paper node,
        return sorted (index, score) pairs, descending."""
        self.eval()
        emb_dict = self.encode(x_dict, edge_index_dict)
        emb_dst = emb_dict[self.dst_type][dst_index].unsqueeze(0)
        emb_dst = emb_dst.expand(len(candidate_src_index), -1)
        emb_src = emb_dict[self.src_type][candidate_src_index]
        scores = self.scorer(emb_src, emb_dst)
        order = torch.argsort(scores, descending=True)
        ranked_idx = candidate_src_index[order]
        ranked_scores = scores[order]
        return list(zip(ranked_idx.tolist(), ranked_scores.tolist()))
    

    @torch.no_grad()
    def rank_experts(self, x_dict, edge_index_dict, dst_index,
                      candidate_src_index):
        """Score all candidate authors against one target venue/paper node,
        return sorted (index, score) pairs, descending."""
        self.eval()
        emb_dict = self.encode(x_dict, edge_index_dict)
        emb_dst = emb_dict[self.dst_type][dst_index].unsqueeze(0)
        emb_dst = emb_dst.expand(len(candidate_src_index), -1)
        emb_src = emb_dict[self.src_type][candidate_src_index]
        scores = self.scorer(emb_src, emb_dst)
        order = torch.argsort(scores, descending=True)
        ranked_idx = candidate_src_index[order]
        ranked_scores = scores[order]
        return list(zip(ranked_idx.tolist(), ranked_scores.tolist()))

    torch.no_grad()
    def rank_experts(self, x_dict, edge_index_dict, dst_index,
                      candidate_src_index):
        """Score all candidate authors against one target venue/paper node,
        return sorted (index, score) pairs, descending."""
        self.eval()
        emb_dict = self.encode(x_dict, edge_index_dict)
        emb_dst = emb_dict[self.dst_type][dst_index].unsqueeze(0)
        emb_dst = emb_dst.expand(len(candidate_src_index), -1)
        emb_src = emb_dict[self.src_type][candidate_src_index]
        scores = self.scorer(emb_src, emb_dst)
        order = torch.argsort(scores, descending=True)
        ranked_idx = candidate_src_index[order]
        ranked_scores = scores[order]
        return list(zip(ranked_idx.tolist(), ranked_scores.tolist()))