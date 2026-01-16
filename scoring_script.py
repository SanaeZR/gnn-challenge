import sys
import pandas as pd
from sklearn.metrics import f1_score


def main():
    if len(sys.argv) != 2:
        print("Usage: python scoring_script.py submissions/your_file.csv")
        sys.exit(1)

    submission_file = sys.argv[1]
    sub = pd.read_csv(submission_file)
    truth = pd.read_csv("data/test_labels.csv")

    if {"id", "target"}.issubset(sub.columns):
        pass
    elif {"node_id", "label"}.issubset(sub.columns):
        sub = sub.rename(columns={"node_id": "id", "label": "target"})
    elif {"id", "label"}.issubset(sub.columns):
        sub = sub.rename(columns={"label": "target"})
    elif {"node_id", "target"}.issubset(sub.columns):
        sub = sub.rename(columns={"node_id": "id"})
    else:
        raise ValueError("Submission must contain columns: id/target (or node_id/label)")

    merged = truth.merge(sub, on="id", suffixes=("_true", "_pred"))
    score = f1_score(merged["target_true"], merged["target_pred"], average="macro")
    print(f"Submission Macro-F1 Score: {score:.4f}")


if __name__ == "__main__":
    main()
