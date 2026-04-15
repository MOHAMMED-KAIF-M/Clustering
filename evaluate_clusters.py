from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_expected_labels(path: Path) -> tuple[dict[str, str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", payload)
    if not isinstance(labels, dict):
        raise ValueError("Expected labels file must be a JSON object or contain a top-level 'labels' object.")
    noise_payload = payload.get("noise", [])
    if not isinstance(noise_payload, list):
        raise ValueError("Expected noise entries must be a list when provided.")
    return {str(image): str(label) for image, label in labels.items()}, {str(image) for image in noise_payload}


def load_clusters(path: Path) -> tuple[dict[str, int], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predicted: dict[str, int] = {}
    for cluster in payload.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        for image_name in cluster.get("images", []):
            predicted[str(image_name)] = cluster_id
    noise = {str(image_name) for image_name in payload.get("noise", [])}
    for image_name in noise:
        predicted.setdefault(image_name, -1)
    return predicted, noise


def expected_label_for(image_name: str, expected: dict[str, str], expected_noise: set[str]) -> str | None:
    if image_name in expected_noise:
        return f"__noise__:{image_name}"
    return expected.get(image_name)


def pairwise_metrics(expected: dict[str, str], expected_noise: set[str], predicted: dict[str, int]) -> dict[str, float]:
    expected_images = set(expected) | set(expected_noise)
    images = sorted(expected_images & set(predicted))
    tp = fp = fn = tn = 0
    for left_index, left_name in enumerate(images):
        for right_name in images[left_index + 1 :]:
            expected_same = (
                left_name not in expected_noise
                and right_name not in expected_noise
                and expected.get(left_name) is not None
                and expected.get(left_name) == expected.get(right_name)
            )
            predicted_same = predicted[left_name] != -1 and predicted[left_name] == predicted[right_name]
            if expected_same and predicted_same:
                tp += 1
            elif not expected_same and predicted_same:
                fp += 1
            elif expected_same and not predicted_same:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    return {
        "pairwise_precision": round(precision, 4),
        "pairwise_recall": round(recall, 4),
        "pairwise_f1": round(f1, 4),
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "true_negative_pairs": tn,
    }


def cluster_purity(expected: dict[str, str], expected_noise: set[str], predicted: dict[str, int]) -> float:
    cluster_members: dict[int, list[str]] = {}
    for image_name, cluster_id in predicted.items():
        if cluster_id == -1 or (image_name not in expected and image_name not in expected_noise):
            continue
        cluster_members.setdefault(cluster_id, []).append(image_name)

    assigned_images = 0
    pure_images = 0
    for members in cluster_members.values():
        labels = Counter(expected_label_for(name, expected, expected_noise) for name in members)
        assigned_images += len(members)
        pure_images += labels.most_common(1)[0][1]
    return round((pure_images / assigned_images) if assigned_images else 0.0, 4)


def cluster_size_stats(predicted: dict[str, int]) -> tuple[int, float]:
    cluster_members: dict[int, int] = {}
    for cluster_id in predicted.values():
        if cluster_id == -1:
            continue
        cluster_members[cluster_id] = cluster_members.get(cluster_id, 0) + 1
    cluster_count = len(cluster_members)
    average_size = (sum(cluster_members.values()) / cluster_count) if cluster_count else 0.0
    return cluster_count, round(average_size, 4)


def evaluate_one(expected_labels: dict[str, str], expected_noise: set[str], clusters_path: Path) -> dict:
    predicted, noise = load_clusters(clusters_path)
    expected_images = set(expected_labels) | set(expected_noise)
    overlap_images = sorted(expected_images & set(predicted))
    labeled_noise = sum(1 for image_name in overlap_images if predicted[image_name] == -1)
    cluster_count, average_cluster_size = cluster_size_stats(predicted)
    metrics = pairwise_metrics(expected_labels, expected_noise, predicted)
    metrics.update(
        {
            "cluster_purity": cluster_purity(expected_labels, expected_noise, predicted),
            "evaluated_images": len(overlap_images),
            "noise_rate": round((labeled_noise / len(overlap_images)) if overlap_images else 0.0, 4),
            "cluster_file": str(clusters_path),
            "predicted_cluster_count": cluster_count,
            "average_cluster_size": average_cluster_size,
            "predicted_noise_count": len(noise),
            "expected_noise_count": len(expected_noise & set(overlap_images)),
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clustering outputs against expected labels.")
    parser.add_argument("--labels", required=True, help="Path to expected labels JSON.")
    parser.add_argument(
        "--clusters",
        nargs="+",
        required=True,
        help="One or more clusters.json files to evaluate and compare.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON file for writing the evaluation report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_labels, expected_noise = load_expected_labels(Path(args.labels))
    report = {
        "labels_file": str(Path(args.labels).resolve()),
        "runs": [evaluate_one(expected_labels, expected_noise, Path(cluster_path).resolve()) for cluster_path in args.clusters],
    }

    for run in report["runs"]:
        print(
            f"{run['cluster_file']}: "
            f"precision={run['pairwise_precision']:.4f} "
            f"recall={run['pairwise_recall']:.4f} "
            f"f1={run['pairwise_f1']:.4f} "
            f"purity={run['cluster_purity']:.4f} "
            f"noise_rate={run['noise_rate']:.4f} "
            f"clusters={run['predicted_cluster_count']} "
            f"avg_cluster_size={run['average_cluster_size']:.4f}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
