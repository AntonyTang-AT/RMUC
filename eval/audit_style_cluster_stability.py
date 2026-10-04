"""Check whether causal early-observation style clusters are stable by match fold."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from audit_early_style_cluster_prior import load_early_profiles  # noqa: E402


def run() -> dict:
    data = load_early_profiles("train")
    keys = list(data["profiles"])
    x = np.vstack([data["profiles"][key] for key in keys])
    games = np.asarray([key[0] for key in keys])
    results = {}

    for n_clusters in (2, 3):
        assignments = []
        fold_cluster_sizes = []
        fold_silhouette = []
        for fold, (train_ix, _heldout_ix) in enumerate(
            GroupKFold(n_splits=5).split(x, groups=games)
        ):
            mean = x[train_ix].mean(axis=0)
            scale = x[train_ix].std(axis=0)
            scale[scale <= 1e-6] = 1.0
            train_x = (x[train_ix] - mean) / scale
            all_x = (x - mean) / scale
            model = KMeans(n_clusters=n_clusters, n_init=20, random_state=100 + fold)
            train_labels = model.fit_predict(train_x)
            assignments.append(model.predict(all_x))
            fold_cluster_sizes.append(np.bincount(train_labels, minlength=n_clusters).tolist())
            fold_silhouette.append(
                float(
                    silhouette_score(
                        train_x,
                        train_labels,
                        sample_size=min(500, len(train_ix)),
                        random_state=fold,
                    )
                )
            )

        ari = [
            adjusted_rand_score(assignments[i], assignments[j])
            for i in range(len(assignments))
            for j in range(i + 1, len(assignments))
        ]
        full_mean = x.mean(axis=0)
        full_scale = x.std(axis=0)
        full_scale[full_scale <= 1e-6] = 1.0
        full_x = (x - full_mean) / full_scale
        full_model = KMeans(n_clusters=n_clusters, n_init=20, random_state=29).fit(full_x)
        counts = np.bincount(full_model.labels_, minlength=n_clusters)
        centroids = np.vstack(
            [x[full_model.labels_ == label].mean(axis=0) for label in range(n_clusters)]
        )
        centroid_differences = []
        for left in range(n_clusters):
            for right in range(left + 1, n_clusters):
                delta = centroids[right] - centroids[left]
                largest = np.argsort(np.abs(delta))[::-1][:12]
                centroid_differences.append(
                    {
                        "clusters": [left, right],
                        "counts": [int(counts[left]), int(counts[right])],
                        "largest_raw_feature_differences": [
                            [int(i), round(float(delta[i]), 6)] for i in largest
                        ],
                    }
                )
        results[f"K{n_clusters}"] = {
            "cross_fold_assignment_ari_median_q25_q75": np.quantile(
                ari, [0.5, 0.25, 0.75]
            ).tolist(),
            "pairwise_assignment_ari": ari,
            "fold_train_silhouette": fold_silhouette,
            "fold_train_cluster_sizes": fold_cluster_sizes,
            "all_train_cluster_sizes": counts.tolist(),
            "largest_raw_centroid_differences": centroid_differences,
        }

    return {
        "train_profiles": len(keys),
        "train_games": int(len(np.unique(games))),
        "fit": "5-fold GroupKFold by game; scaler and KMeans fit on four folds only",
        "assignment_comparison": "each fold-fitted model assigns the same causal training profiles; ARI is permutation-invariant",
        "target_labels_used": False,
        "validation_or_test_labels_used": False,
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=True, indent=2))
