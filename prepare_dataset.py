import dgl
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_cora_challenge():
    """
    Prépare le dataset Cora pour un challenge de node classification
    """
    print("Téléchargement du dataset Cora...")

    # Charger le dataset Cora depuis DGL

    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]

    # Extraire les caractéristiques et labels
    features = g.ndata['feat'].numpy()
    labels = g.ndata['label'].numpy()

    # Créer un DataFrame avec les features
    feature_cols = [f'feature_{i}' for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    df['node_id'] = range(len(df))

    # Extraire les arêtes du graphe
    src, dst = g.edges()
    edges_df = pd.DataFrame({
        'source': src.numpy(),
        'target': dst.numpy()
    })

    # Split train/test (80/20)
    train_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Créer train.csv (avec labels)
    train_df = df.iloc[train_idx].copy()
    train_df.to_csv('data/train.csv', index=False)
    print(f"✓ train.csv créé : {len(train_df)} nœuds")

    # Créer test.csv (sans labels)
    test_df = df.iloc[test_idx].copy()
    test_df_no_labels = test_df.drop('label', axis=1)
    test_df_no_labels.to_csv('data/test.csv', index=False)
    print(f"✓ test.csv créé : {len(test_df)} nœuds")

    # Sauvegarder les vraies labels pour l'évaluation
    test_df[['node_id', 'label']].to_csv('data/test_labels_hidden.csv', index=False)
    print("✓ test_labels_hidden.csv créé (pour l'évaluation)")

    # Sauvegarder le graphe (arêtes)
    edges_df.to_csv('data/edges.csv', index=False)
    print(f"✓ edges.csv créé : {len(edges_df)} arêtes")

    # Créer un fichier de submission exemple
    sample_submission = test_df[['node_id']].copy()
    sample_submission['label'] = 0  # Prédictions par défaut
    sample_submission.to_csv('submissions/sample_submission.csv', index=False)
    print("✓ sample_submission.csv créé")

    # Afficher les statistiques
    print("\n=== Statistiques du Dataset ===")
    print(f"Nombre total de nœuds : {len(df)}")
    print(f"Nombre de features : {features.shape[1]}")
    print(f"Nombre de classes : {len(np.unique(labels))}")
    print(f"Nombre d'arêtes : {len(edges_df)}")
    print(f"Distribution des classes :")
    print(df['label'].value_counts().sort_index())

if __name__ == "__main__":
    prepare_cora_challenge()
