import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def load_graph(edges_path, num_nodes):
    edges = pd.read_csv(edges_path)
    if {"src", "dst"}.issubset(edges.columns):
        src = edges["src"].values
        dst = edges["dst"].values
    elif {"source", "target"}.issubset(edges.columns):
        src = edges["source"].values
        dst = edges["target"].values
    else:
        raise ValueError("Edges file must contain columns 'src'/'dst' or 'source'/'target'.")
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g = dgl.to_bidirected(g)
    g = dgl.to_simple(g)
    g = dgl.add_self_loop(g)
    return g


def get_feature_cols(df):
    cols = [c for c in df.columns if c.startswith("f") or c.startswith("feature_")]
    if not cols:
        raise ValueError("No feature columns found (expected prefix 'f' or 'feature_').")
    return cols


def get_label_col(df):
    if "target" in df.columns:
        return "target"
    if "label" in df.columns:
        return "label"
    raise ValueError("No label column found (expected 'target' or 'label').")


def get_id_col(df):
    if "id" in df.columns:
        return "id"
    if "node_id" in df.columns:
        return "node_id"
    raise ValueError("No id column found (expected 'id' or 'node_id').")


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        self.conv1 = GINConv(mlp1, "sum")
        self.conv2 = GINConv(mlp2, "sum")
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h


def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    feature_cols = get_feature_cols(train_df)
    label_col = get_label_col(train_df)
    id_col = get_id_col(train_df if "node_id" in train_df.columns or "id" in train_df.columns else test_df)

    train_ids = train_df[id_col].values.astype(np.int64)
    test_ids = test_df[get_id_col(test_df)].values.astype(np.int64)
    num_nodes = int(max(train_ids.max(initial=0), test_ids.max(initial=0)) + 1)

    x_full = torch.zeros((num_nodes, len(feature_cols)), dtype=torch.float32)
    x_full[torch.tensor(train_ids)] = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
    x_full[torch.tensor(test_ids)] = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)

    y_full = torch.full((num_nodes,), -1, dtype=torch.long)
    y_full[torch.tensor(train_ids)] = torch.tensor(train_df[label_col].values, dtype=torch.long)

    g = load_graph(os.path.join(data_dir, "edges.csv"), num_nodes=num_nodes)

    train_ids = train_ids.astype(np.int64)
    y_train = y_full[torch.tensor(train_ids)].numpy()
    train_idx, val_idx = train_test_split(
        train_ids, test_size=0.2, stratify=y_train, random_state=42
    )

    model = GIN(in_dim=x_full.shape[1], hidden_dim=64, num_classes=len(np.unique(y_train)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    best_val = 0.0
    for epoch in range(1, 101):
        model.train()
        logits = model(g, x_full)
        loss = F.cross_entropy(logits[train_idx], y_full[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(g, x_full)
            val_pred = val_logits[val_idx].argmax(dim=1)
            val_f1 = f1_score(y_full[val_idx].numpy(), val_pred.numpy(), average="macro")

        if val_f1 > best_val:
            best_val = val_f1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val Macro-F1 {val_f1:.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(g, x_full)
        test_pred = test_logits[test_ids].argmax(dim=1).numpy()
    out_label_col = "target" if label_col == "target" else "label"
    sub = pd.DataFrame({get_id_col(test_df): test_ids, out_label_col: test_pred})
    out_path = os.path.join(base_dir, "..", "submissions", "sample_submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved sample submission to {out_path}")


if __name__ == "__main__":
    main()
