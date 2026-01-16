import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def compute_degree(edges_path, num_nodes):
    edges = pd.read_csv(edges_path)
    deg = np.zeros(num_nodes, dtype=np.int64)
    for src, dst in zip(edges["src"].values, edges["dst"].values):
        deg[src] += 1
        deg[dst] += 1
    return deg


def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    feature_cols = [c for c in train_df.columns if c.startswith("f")]

    train_deg = compute_degree(os.path.join(data_dir, "train_edges.csv"), num_nodes=len(train_df))
    test_deg = compute_degree(os.path.join(data_dir, "test_edges.csv"), num_nodes=len(test_df))

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["deg"] = train_deg
    test_df["deg"] = test_deg

    X = train_df[feature_cols + ["deg"]].values
    y = train_df["target"].values
    X_test = test_df[feature_cols + ["deg"]].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    print(f"Validation Macro-F1: {val_f1:.4f}")

    clf.fit(X, y)
    test_pred = clf.predict(X_test)

    sub = pd.DataFrame({"id": test_df["id"].values, "target": test_pred})
    out_path = os.path.join(base_dir, "..", "submissions", "sample_submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved sample submission to {out_path}")


if __name__ == "__main__":
    main()
