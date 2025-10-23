#!/usr/bin/env python3
"""
Train a Variational Graph Autoencoder (VGAE) on a chromatin contact graph.

Inputs:
  - A PyTorch Geometric Data object saved with torch.save(...) containing:
        x            : [num_nodes, num_features] node features
        edge_index   : [2, num_edges] undirected edges (will be coalesced)
        edge_weight  : [num_edges]   (optional, unused by VGAE)

  - from build_graph.py
---        
Outputs (under results/):
  - model.pt            : trained VGAE state_dict
  - emb.npy             : node embeddings (mean; shape [num_nodes, latent_dim])
  - metrics.json        : train/val/test AUC/AP summary
"""

import os, json, argparse, numpy as np, torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import VGAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


class Encoder(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int, dropout: float = 0.2):
        super().__init__()
        self.gc1 = GCNConv(in_dim, hidden, add_self_loops=True, normalize=True)
        self.gc_mu = GCNConv(hidden, latent, add_self_loops=True, normalize=True)
        self.gc_log = GCNConv(hidden, latent, add_self_loops=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.gc1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.gc_mu(h, edge_index), self.gc_log(h, edge_index)


@torch.no_grad()
def eval_linkpred(model, data_like, z):
    """Compute AUROC/AP using provided positive/negative edges."""
    pos = data_like.pos_edge_index
    neg = data_like.neg_edge_index
    # model.test returns (auc, ap) but relies on torchmetrics in some versions;
    # compute explicitly for stability:
    def sigmoid(x): return 1 / (1 + torch.exp(-x))

    # Inner product decoder scores
    def scores(edges):
        src, dst = edges
        s = (z[src] * z[dst]).sum(dim=1)
        return sigmoid(s).cpu().numpy()

    y_true = np.concatenate([np.ones(pos.size(1)), np.zeros(neg.size(1))])
    y_pred = np.concatenate([scores(pos), scores(neg)])

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return auc, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="Path to Data .pt file")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load graph ----
    data = torch.load(args.graph)
    # Coalesce/clean edges
    ei, _ = remove_self_loops(data.edge_index)
    data.edge_index = to_undirected(ei, num_nodes=data.num_nodes)
    x = data.x.float()

    # ---- Split edges for link prediction ----
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        split_labels=False,
    )
    train_data, val_data, test_data = splitter(data)

    # Positive edges are just the edges in each split
    train_data.pos_edge_index = train_data.edge_index
    val_data.pos_edge_index = val_data.edge_index
    test_data.pos_edge_index = test_data.edge_index

    # Generate negative edges for validation and test manually
    for subset in [val_data, test_data]:
        subset.neg_edge_index = negative_sampling(
            edge_index=subset.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=subset.edge_index.size(1),
            method='sparse'
        )


    # ---- Model ----
    enc = Encoder(in_dim=x.size(1), hidden=args.hidden, latent=args.latent, dropout=args.dropout)
    model = VGAE(enc)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Training loop ----
    best_val_auc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        # Encode using remaining training edges
        z = model.encode(x, train_data.edge_index)
        # Reconstruction loss on positive training edges (negatives sampled inside)
        loss_recon = model.recon_loss(z, train_data.pos_edge_index)
        # KL divergence regularizer
        loss_kl = (1.0 / data.num_nodes) * model.kl_loss()
        loss = loss_recon + loss_kl
        loss.backward()
        optimizer.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            z_full = model.encode(x, data.edge_index)  # use full graph for eval embeddings
            val_auc, val_ap = eval_linkpred(model, val_data, z_full)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:03d}/{args.epochs}] loss={loss.item():.4f} | val AUC={val_auc:.4f} AP={val_ap:.4f}")

    # ---- Save best model ----
    model.load_state_dict(best_state)
    model_path = os.path.join(args.outdir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # ---- Final test metrics ----
    model.eval()
    with torch.no_grad():
        z_final = model.encode(x, data.edge_index)
        test_auc, test_ap = eval_linkpred(model, test_data, z_final)

    # ---- Save embeddings & metrics ----
    emb_path = os.path.join(args.outdir, "emb.npy")
    np.save(emb_path, z_final.cpu().numpy())

    metrics = {
        "val_auc": float(best_val_auc),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "epochs": args.epochs,
        "hidden": args.hidden,
        "latent": args.latent,
        "dropout": args.dropout,
        "lr": args.lr,
        "seed": args.seed
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model -> {model_path}")
    print(f"Saved embeddings -> {emb_path}  (shape={z_final.shape})")
    print(f"Metrics: AUC(test)={test_auc:.4f}, AP(test)={test_ap:.4f}")


if __name__ == "__main__":
    main()
