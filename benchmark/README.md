# Benchmark Guide

This folder is the starting point for the evaluation harness.

## Files

- `labels_template.json`
  Template for expected image-to-group labels.
- `real_estate_ground_truth.json`
  Real labeled ground truth for the current `input/` image set, including the expected noise image.

## Label Format

```json
{
  "labels": {
    "image_001.jpg": "group_a",
    "image_002.jpg": "group_a",
    "image_003.jpg": "group_b"
  }
}
```

Images with the same label are expected to be in the same cluster.

If your dataset has expected noise images, add them to a top-level `noise` list.

## Evaluate One Or More Runs

```powershell
python evaluate_clusters.py --labels .\benchmark\labels_template.json --clusters .\output\clusters.json
```

Evaluate against the real labeled benchmark:

```powershell
python evaluate_clusters.py --labels .\benchmark\real_estate_ground_truth.json --clusters .\output\clusters.json .\output_strict\clusters.json
```

Compare multiple runs:

```powershell
python evaluate_clusters.py --labels .\benchmark\labels_template.json --clusters .\output\clusters.json .\output_strict\clusters.json
```

Write a JSON comparison report:

```powershell
python evaluate_clusters.py --labels .\benchmark\labels_template.json --clusters .\output\clusters.json .\output_strict\clusters.json --output .\benchmark\comparison_report.json
```

## Reported Metrics

- pairwise precision
- pairwise recall
- pairwise F1
- cluster purity
- noise rate
- predicted cluster count
- average cluster size
