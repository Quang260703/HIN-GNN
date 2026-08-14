"""
HAN (Heterogeneous Attention Network) for related-paper retrieval.

Pipeline:
  1. Load the OpenAlex CSV (paper_id, title, publication_year, citation_count,
     referenced_works_count, referenced_works, related_works, author_ids,
     authors, institutions, source_id, source, fwci).
  2. Build a heterogeneous graph with three node types (paper, author, venue)
     and edge types (author -> writes -> paper, venue -> publishes -> paper,
     paper -> cites -> paper), plus their reverses for message passing.
  3. Sort papers by publication date and split 80/20 (train = earliest 80%,
     test = most recent 20%) so evaluation never leaks future citation info
     into training.
  4. Train a HAN encoder with a weighted BPR loss on citation edges within
     the train window.
  5. For a query paper, embed all candidate papers published strictly before
     it and return the top-10 by cosine similarity.

Requires: torch, torch_geometric, pandas, numpy
    pip install torch torch_geometric pandas numpy --break-system-packages
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv


# --------------------------------------------------------------------------
# 1. Data loading + graph construction
# --------------------------------------------------------------------------

def short_id(openalex_url: str) -> str:
    """https://openalex.org/W123 -> W123 (also handles empty/NaN safely)."""
    if not isinstance(openalex_url, str) or not openalex_url:
        return ""
    return openalex_url.rstrip("/").split("/")[-1]


def split_list(cell: str) -> List[str]:
    if not isinstance(cell, str) or not cell:
        return []
    return [short_id(x) for x in cell.split(";") if x]


@dataclass
class GraphBundle:
    data: HeteroData
    paper_index: Dict[str, int]          # paper_id -> node index
    index_paper: List[str]               # node index -> paper_id
    pub_year: np.ndarray                 # per-paper publication year, aligned to paper_index
    pub_order: np.ndarray                # argsort of papers by time (earliest first)
    train_mask: np.ndarray               # bool, len = num papers
    test_mask: np.ndarray
    related_works: Dict[int, set] = field(default_factory=dict)
    # ground-truth labels for evaluation ONLY, keyed by paper node index -> set
    # of node indices from that paper's `related_works` field that are also
    # present in this corpus. Never used to build graph edges or features —
    # using it there would leak the evaluation label into training.


def build_graph(csv_path: str, val_frac: float = 0.0) -> GraphBundle:
    df = pd.read_csv(csv_path)
    df["paper_id_short"] = df["paper_id"].apply(short_id)
    df = df.drop_duplicates(subset="paper_id_short").reset_index(drop=True)

    # ---- node index maps ----
    paper_index = {pid: i for i, pid in enumerate(df["paper_id_short"])}
    index_paper = list(df["paper_id_short"])

    author_set, venue_set = set(), set()
    df["author_list"] = df["author_ids"].apply(split_list)
    df["venue_short"] = df["source_id"].apply(short_id)
    for authors in df["author_list"]:
        author_set.update(authors)
    venue_set.update(v for v in df["venue_short"] if v)

    author_index = {a: i for i, a in enumerate(sorted(author_set))}
    venue_index = {v: i for i, v in enumerate(sorted(venue_set))}

    # ---- paper node features (non-textual / structural, per project design) ----
    def safe_float(x, default=0.0):
        try:
            v = float(x)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    citation_count = df["citation_count"].apply(safe_float).to_numpy()
    ref_count = df["referenced_works_count"].apply(safe_float).to_numpy()
    fwci = df["fwci"].apply(safe_float).to_numpy()
    pub_year = df["publication_year"].apply(lambda y: safe_float(y, 2000)).to_numpy()

    def zscore(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sigma = x.mean(), x.std() + 1e-6
        return (x - mu) / sigma

    # NOTE: fwci sometimes contains clearly corrupted values (e.g. thousands,
    # which is impossible for a field-normalized score that should center on
    # ~1.0). Clip before z-scoring so one bad row doesn't dominate the scale.
    fwci_clipped = np.clip(fwci, 0, 25)

    paper_feats = np.stack(
        [zscore(citation_count), zscore(ref_count), zscore(fwci_clipped), zscore(pub_year)],
        axis=1,
    ).astype(np.float32)

    num_authors = len(author_index)
    num_venues = len(venue_index)
    # Authors/venues have no numeric metadata in this schema, so give them a
    # learnable embedding instead of a hand-built feature vector; HANConv
    # will project whatever dimension you hand it, so a small placeholder
    # dimension is fine as an nn.Embedding lookup at model init time.
    author_feats = np.zeros((num_authors, 1), dtype=np.float32)
    venue_feats = np.zeros((num_venues, 1), dtype=np.float32)

    # ---- edges ----
    writes_src, writes_dst = [], []          # author -> paper
    publishes_src, publishes_dst = [], []    # venue -> paper
    cites_src, cites_dst = [], []            # paper -> paper (citing -> cited)

    for row in df.itertuples(index=False):
        pid = row.paper_id_short
        p_idx = paper_index[pid]

        for a in row.author_list:
            writes_src.append(author_index[a])
            writes_dst.append(p_idx)

        v = row.venue_short
        if v and v in venue_index:
            publishes_src.append(venue_index[v])
            publishes_dst.append(p_idx)

        for ref in split_list(row.referenced_works):
            if ref in paper_index:              # only keep edges within this corpus
                cites_src.append(p_idx)
                cites_dst.append(paper_index[ref])

    data = HeteroData()
    data["paper"].x = torch.tensor(paper_feats)
    data["author"].x = torch.tensor(author_feats)
    data["venue"].x = torch.tensor(venue_feats)

    def edge_index(src, dst):
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    data["author", "writes", "paper"].edge_index = edge_index(writes_src, writes_dst)
    data["paper", "written_by", "author"].edge_index = edge_index(writes_dst, writes_src)
    data["venue", "publishes", "paper"].edge_index = edge_index(publishes_src, publishes_dst)
    data["paper", "published_in", "venue"].edge_index = edge_index(publishes_dst, publishes_src)
    data["paper", "cites", "paper"].edge_index = edge_index(cites_src, cites_dst)
    data["paper", "cited_by", "paper"].edge_index = edge_index(cites_dst, cites_src)

    # ---- related_works: evaluation ground truth ONLY (not a graph edge, not a feature) ----
    related_works: Dict[int, set] = {}
    for row in df.itertuples(index=False):
        p_idx = paper_index[row.paper_id_short]
        gt = {
            paper_index[rid]
            for rid in split_list(row.related_works)
            if rid in paper_index and rid != row.paper_id_short
        }
        if gt:
            related_works[p_idx] = gt

    # ---- time-sorted 80/20 split (earliest 80% train, most recent 20% test) ----
    pub_order = np.argsort(pub_year, kind="stable")
    n = len(pub_order)
    n_train = int(round(n * (1 - val_frac) * 0.8)) if val_frac == 0 else int(round(n * 0.8))
    train_ids = set(pub_order[:n_train].tolist())
    train_mask = np.array([i in train_ids for i in range(n)], dtype=bool)
    test_mask = ~train_mask

    return GraphBundle(
        data=data,
        paper_index=paper_index,
        index_paper=index_paper,
        pub_year=pub_year,
        pub_order=pub_order,
        train_mask=train_mask,
        test_mask=test_mask,
        related_works=related_works,
    )


# --------------------------------------------------------------------------
# 2. HAN model
# --------------------------------------------------------------------------

class HANEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim=64, out_dim=64, heads=4, dropout=0.2):
        super().__init__()
        self.han = HANConv(
            in_channels=-1,          # lazy init per node type
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            metadata=metadata,
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        out = self.han(x_dict, edge_index_dict)
        # HANConv returns None for node types it couldn't reach this layer;
        # guard against that when computing the paper projection.
        paper_h = out["paper"]
        return self.proj(F.elu(paper_h))


# --------------------------------------------------------------------------
# 3. Weighted BPR loss over citation edges
#    weight = a simple closeness proxy (shared-reference overlap count),
#    matching the project's role/path-weighted closeness idea. Swap in the
#    Weight1-6 path-based scheme here if you've already implemented it
#    elsewhere in the pipeline.
# --------------------------------------------------------------------------

def citation_closeness_weight(edge_index: torch.Tensor, num_papers: int) -> torch.Tensor:
    if edge_index.size(1) == 0:
        return torch.empty(0)
    src, dst = edge_index
    # crude proxy: weight by (out-degree of src normalized) so highly-citing
    # survey-like papers don't dominate the loss disproportionately
    out_deg = torch.zeros(num_papers)
    out_deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    w = 1.0 / torch.clamp(out_deg[src], min=1.0)
    return w / w.mean().clamp(min=1e-6)


def sample_negatives(num_papers: int, num_samples: int, exclude_idx: torch.Tensor) -> torch.Tensor:
    neg = torch.randint(0, num_papers, (num_samples,))
    # simple rejection pass for the rare collision with the positive index
    clash = neg == exclude_idx
    while clash.any():
        neg[clash] = torch.randint(0, num_papers, (int(clash.sum()),))
        clash = neg == exclude_idx
    return neg


def weighted_bpr_loss(query_emb, pos_emb, neg_emb, weights, margin=1.0):
    pos_score = (query_emb * pos_emb).sum(-1)
    neg_score = (query_emb * neg_emb).sum(-1)
    # margin scaled by the normalized closeness weight, as decided for this project
    scaled_margin = margin * weights
    return F.softplus(-(pos_score - neg_score) + scaled_margin).mean()


# --------------------------------------------------------------------------
# 4. Training loop
# --------------------------------------------------------------------------

def train(bundle: GraphBundle, epochs=50, lr=1e-3, device="cpu"):
    data = bundle.data.to(device)
    model = HANEncoder(metadata=data.metadata()).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    cite_ei = data["paper", "cites", "paper"].edge_index
    num_papers = data["paper"].x.size(0)

    train_paper_idx = torch.tensor(np.where(bundle.train_mask)[0])
    train_edge_mask = torch.isin(cite_ei[0], train_paper_idx) & torch.isin(cite_ei[1], train_paper_idx)
    train_cite_ei = cite_ei[:, train_edge_mask]
    edge_weights = citation_closeness_weight(train_cite_ei, num_papers).to(device)

    if train_cite_ei.size(1) == 0:
        raise ValueError(
            "No citation edges fall inside the train window — check that "
            "referenced_works actually point to other papers present in this CSV."
        )

    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        paper_emb = model(data.x_dict, data.edge_index_dict)

        src, pos = train_cite_ei
        neg = sample_negatives(num_papers, src.size(0), pos).to(device)

        loss = weighted_bpr_loss(
            paper_emb[src], paper_emb[pos], paper_emb[neg], edge_weights
        )
        loss.backward()
        optim.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

    return model, data


# --------------------------------------------------------------------------
# 5. Top-10 retrieval for a query paper
# --------------------------------------------------------------------------

@torch.no_grad()
def top_k_related(model, data, bundle: GraphBundle, query_paper_id: str, k: int = 10, device="cpu"):
    model.eval()
    paper_emb = model(data.x_dict, data.edge_index_dict)

    q_short = short_id(query_paper_id) if query_paper_id.startswith("http") else query_paper_id
    if q_short not in bundle.paper_index:
        raise KeyError(f"{query_paper_id} not found in this corpus.")
    q_idx = bundle.paper_index[q_short]
    q_year = bundle.pub_year[q_idx]

    # restrict candidates to papers published strictly before the query,
    # matching the project's evaluation protocol
    candidate_mask = bundle.pub_year < q_year
    candidate_mask[q_idx] = False
    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) == 0:
        return []

    q_vec = F.normalize(paper_emb[q_idx : q_idx + 1], dim=-1)
    cand_vecs = F.normalize(paper_emb[candidate_idx], dim=-1)
    sims = (q_vec @ cand_vecs.T).squeeze(0)

    top = torch.topk(sims, k=min(k, len(candidate_idx)))
    results = []
    for score, local_i in zip(top.values.tolist(), top.indices.tolist()):
        p_idx = candidate_idx[local_i]
        results.append((bundle.index_paper[p_idx], score))
    return results


# --------------------------------------------------------------------------
# 6. Precision@10 / Recall@10 on the test split
#    Ground truth = each test paper's `related_works` field, restricted to
#    papers present in this corpus. This field is used ONLY here, for
#    scoring — it never touches graph construction, features, or training.
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_precision_recall(model, data, bundle: GraphBundle, k: int = 10, device="cpu"):
    model.eval()
    paper_emb = model(data.x_dict, data.edge_index_dict)

    test_idx = np.where(bundle.test_mask)[0]
    precisions, recalls = [], []
    skipped_no_gt = 0

    for p_idx in test_idx:
        gt = bundle.related_works.get(int(p_idx))
        if not gt:
            skipped_no_gt += 1
            continue  # can't score precision/recall with no ground truth

        q_year = bundle.pub_year[p_idx]
        candidate_mask = bundle.pub_year < q_year
        candidate_mask[p_idx] = False
        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            continue

        q_vec = F.normalize(paper_emb[p_idx : p_idx + 1], dim=-1)
        cand_vecs = F.normalize(paper_emb[candidate_idx], dim=-1)
        sims = (q_vec @ cand_vecs.T).squeeze(0)

        top = torch.topk(sims, k=min(k, len(candidate_idx)))
        retrieved = {int(candidate_idx[i]) for i in top.indices.tolist()}

        hits = len(retrieved & gt)
        precisions.append(hits / k)                 # precision@10 (denominator fixed at k)
        recalls.append(hits / len(gt))               # recall@10 (denominator = |ground truth|)

    if not precisions:
        raise ValueError(
            "No test papers had usable ground truth / candidates — check that "
            "related_works IDs actually overlap with papers in this corpus."
        )

    mean_p, mean_r = float(np.mean(precisions)), float(np.mean(recalls))
    print(
        f"\nEvaluated on {len(precisions)} test papers "
        f"({skipped_no_gt} skipped: no related_works overlap in corpus)."
    )
    print(f"Precision@{k}: {mean_p:.4f}")
    print(f"Recall@{k}:    {mean_r:.4f}")
    return mean_p, mean_r


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to the OpenAlex works CSV")
    ap.add_argument("--query", required=True, help="paper_id (short or full OpenAlex URL) to retrieve related papers for")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-eval", action="store_true", help="skip Precision@k/Recall@k on the test split")
    args = ap.parse_args()

    bundle = build_graph(args.csv_path)
    print(f"Loaded {len(bundle.index_paper)} papers "
          f"({bundle.train_mask.sum()} train / {bundle.test_mask.sum()} test, time-sorted 80/20).")

    model, data = train(bundle, epochs=args.epochs, device=args.device)

    results = top_k_related(model, data, bundle, args.query, k=args.k, device=args.device)
    print(f"\nTop-{args.k} related papers for {args.query}:")
    for rank, (pid, score) in enumerate(results, 1):
        print(f"{rank:2d}. {pid}   score={score:.4f}")

    if not args.skip_eval:
        evaluate_precision_recall(model, data, bundle, k=args.k, device=args.device)


if __name__ == "__main__":
    main()
