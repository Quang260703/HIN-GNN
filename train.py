import argparse
import random

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model import ExpertRetrievalModel


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def sample_negative_edges(num_src, num_dst, num_samples, exclude_set=None,
                           device="cpu"):
    """Uniform negative sampling of (src, dst) pairs not in exclude_set."""
    exclude_set = exclude_set or set()
    src, dst = [], []
    while len(src) < num_samples:
        s = random.randrange(num_src)
        d = random.randrange(num_dst)
        if (s, d) in exclude_set:
            continue
        src.append(s)
        dst.append(d)
    return (torch.tensor(src, device=device),
            torch.tensor(dst, device=device))


def split_edges(edge_index, val_frac=0.1, test_frac=0.1, seed=42):
    """edge_index: LongTensor [2, num_edges] of (src, dst) pairs.
    Returns train/val/test edge_index tensors."""
    g = torch.Generator().manual_seed(seed)
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges, generator=g)
    n_val = int(num_edges * val_frac)
    n_test = int(num_edges * test_frac)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return edge_index[:, train_idx], edge_index[:, val_idx], edge_index[:, test_idx]


@torch.no_grad()
def evaluate(model, x_dict, edge_index_dict, pos_edges, num_src, num_dst,
             device):
    model.eval()
    neg_src, neg_dst = sample_negative_edges(
        num_src, num_dst, pos_edges.size(1), device=device
    )
    pos_scores = model(x_dict, edge_index_dict, pos_edges[0], pos_edges[1])
    neg_scores = model(x_dict, edge_index_dict, neg_src, neg_dst)

    scores = torch.cat([pos_scores, neg_scores]).sigmoid().cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0)),
    ]).numpy()
    return roc_auc_score(labels, scores)


def train_one_epoch(model, optimizer, x_dict, edge_index_dict, train_edges,
                     num_src, num_dst, device, neg_ratio=1):
    model.train()
    optimizer.zero_grad()

    pos_src, pos_dst = train_edges[0], train_edges[1]
    num_neg = pos_src.size(0) * neg_ratio
    neg_src, neg_dst = sample_negative_edges(num_src, num_dst, num_neg,
                                              device=device)

    src_index = torch.cat([pos_src, neg_src])
    dst_index = torch.cat([pos_dst, neg_dst])
    labels = torch.cat([
        torch.ones(pos_src.size(0), device=device),
        torch.zeros(neg_src.size(0), device=device),
    ])

    logits = model(x_dict, edge_index_dict, src_index, dst_index)
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                         help="Path to a saved HeteroData object (torch.save)")
    parser.add_argument("--encoder", type=str, default="hgt",
                         choices=["hgt", "han", "rgcn"])
    parser.add_argument("--src_type", type=str, default="author")
    parser.add_argument("--dst_type", type=str, default="venue")
    parser.add_argument("--edge_key", type=str, default=("author", "writes", "venue"),
                         help="Edge type key in data.edge_index_dict for the "
                              "supervision task (as 'src__rel__dst' string)")
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load graph ---
    data = torch.load(args.data_path)
    data = data.to(device)
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    metadata = data.metadata()

    edge_key = tuple(args.edge_key.split("__")) if isinstance(args.edge_key, str) else args.edge_key
    supervision_edges = data[edge_key].edge_index

    train_edges, val_edges, test_edges = split_edges(supervision_edges)

    num_src = data[args.src_type].num_nodes
    num_dst = data[args.dst_type].num_nodes

    # --- Build model ---
    model = ExpertRetrievalModel(
        encoder_name=args.encoder,
        metadata=metadata,
        src_type=args.src_type,
        dst_type=args.dst_type,
        emb_dim=args.emb_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, x_dict, edge_index_dict,
                                train_edges, num_src, num_dst, device)

        if epoch % 5 == 0 or epoch == args.epochs:
            val_auc = evaluate(model, x_dict, edge_index_dict, val_edges,
                                num_src, num_dst, device)
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | val AUC {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), args.out_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # --- Final test evaluation ---
    model.load_state_dict(torch.load(args.out_path))
    test_auc = evaluate(model, x_dict, edge_index_dict, test_edges,
                         num_src, num_dst, device)
    print(f"Best val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()