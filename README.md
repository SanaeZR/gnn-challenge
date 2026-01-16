# GNN Mini-Competition: Inductive Imbalanced Node Classification

This challenge follows the DGL course (Lectures 1.1–4.6) and focuses on **inductive node classification** with **class imbalance**. You are given a training graph and a separate test graph. Your goal is to predict node labels for the unseen test graph.

## Problem
- **Task:** Inductive node classification on a sparse graph.
- **Why it's hard:** Strong class imbalance and a separate test graph (inductive generalization).
- **Allowed methods:** Message-passing GNNs, sampling-based training, and other methods covered in DGL lectures 1.1–4.6.

## Dataset
Small, synthetic graphs (easy to download, fast to run):
- `data/train.csv`: node features + labels
- `data/test.csv`: node features (labels hidden)
- `data/train_edges.csv`: edges for training graph
- `data/test_edges.csv`: edges for test graph
- `data/test_labels.csv`: hidden ground truth for organizers only

Features are 10‑dimensional numeric vectors. Labels are 3 classes with imbalance.

## Metric
**Macro‑F1** (robust to class imbalance).

## Constraints
1) **No external data**.
2) **Inductive setting**: train only on the training graph, test on the separate test graph.
3) **Runtime**: <= 10 minutes on CPU.

## Repo Structure
```
.
├── data/
├── submissions/
├── starter_code/
├── scoring_script.py
├── leaderboard.md
└── README.md
```

## Baselines
### Baseline (feature + degree, no GNN)
A lightweight baseline is provided in `starter_code/baseline.py`. It uses node features plus graph degree and trains a RandomForest classifier.

### Optional GNN baseline
A GIN baseline (DGL + PyTorch) is provided as `starter_code/baseline_gnn.py` for those with DGL installed.

### Install (baseline)
```
pip install -r starter_code/requirements.txt
```

### Run baseline
```
cd starter_code
python baseline.py
```
This will write a `submissions/sample_submission.csv`.

## Submission Format
`submissions/your_submission.csv`
```
id,target
0,2
1,0
...
```

## Scoring (Organizers)
```
python scoring_script.py submissions/your_submission.csv
```

## Leaderboard
Update `leaderboard.md` with your best Macro‑F1 and model details.

---
Inspired by DGL course materials and the mini‑competition template in `main.pdf`.
