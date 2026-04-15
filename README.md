# CLIP + HDBSCAN Real-Estate Image Clustering

This project clusters property images from an input folder into visually and semantically similar groups.

It is a plain Python script. There is no FastAPI, no Django, and no web service layer.

The current pipeline is designed for cases like:

- kitchen vs hall vs empty room style separation
- same room but different corner/view separation
- broad real-estate image grouping with threshold tuning

## What Is Implemented

- CLIP image embeddings for semantic similarity
- grayscale layout features for room structure
- edge features for geometry cues
- HSV color histogram features
- SSIM-like structural viewpoint scoring behind a tunable weight
- learned local descriptor scoring from CLIP tile crops
- HDBSCAN clustering with tunable thresholds
- second-stage same-corner refinement
- strict same-corner plus same-items mode
- ORB local matching for difficult viewpoint splits
- graph-based clustering alternative for viewpoint and strict refinement
- merge-back of over-split subclusters using semantic and viewpoint compatibility
- explicit noise reasoning plus guarded reassignment
- feature caching to avoid repeated image loading
- config-backed presets and prompt sets
- JSON output plus clustered image folders
- contact sheets, `summary.html`, and `run_manifest.json`
- benchmark evaluation harness with comparison metrics

## What Is Not Implemented Yet

This project does not currently use:

- object detection such as YOLO, DETR, or Faster R-CNN
- explicit scene classifiers such as Places365
- object co-occurrence logic such as `bed + side table`
- occupancy detection such as empty vs furnished as a dedicated model
- luxury or material scoring
- GLCM, LBP, Hough transform, or vanishing-point modeling

That means the system is currently strongest at:

- semantic grouping
- viewpoint-aware grouping
- real-estate photo clustering

and weaker at:

- explicit room labeling from detected objects
- separating visually similar rooms with different hidden semantics

## How The Pipeline Works

The full flow is in [cluster_images.py](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py>).

### 1. Discover input images

The script reads all supported files from the input folder:

- `.jpg`
- `.jpeg`
- `.png`

See [discover_images](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:198>).

### 2. Load images safely

Each image is opened with Pillow, EXIF orientation is normalized, and the image is converted to RGB.

Unreadable or corrupted files are skipped.

See [load_rgb_image](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:209>).

### 3. Extract semantic features with CLIP

Each RGB image is passed through CLIP to generate an embedding vector.

This captures high-level meaning such as:

- kitchen
- empty room
- open-plan area
- similar furniture or scene content

Those embeddings are L2-normalized before clustering.

See [load_clip_module](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:55>) and [embed_images](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:605>).

### 4. Extract visual structure features

For each image, the script also builds extra features:

- layout vector from grayscale image
- edge vector from grayscale edge map
- HSV color histogram
- CLIP prompt-signature features for room/items in strict mode

These features are intended to capture:

- room structure
- wall and opening arrangement
- similar corners and framing
- broad appearance cues
- prompt-level room/item cues in strict mode
- a lightweight structural similarity signal when enabled

See:

- [image_to_layout_vector](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:224>)
- [image_to_edge_vector](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:231>)
- [image_to_color_histogram](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:238>)
- [extract_visual_features](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:552>)

Most image-derived features are cached once per image and reused in refinement, diagnostics, and output generation.

### 5. Fuse semantic and visual features

The pipeline combines:

- CLIP semantic embedding
- layout features
- edge features
- color features

Each block is weighted by CLI thresholds:

- `--semantic-weight`
- `--layout-weight`
- `--edge-weight`
- `--color-weight`

Then the combined feature vector is normalized again.

See [combine_features](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:574>).

### 6. First-stage clustering: semantic grouping

The script first clusters only on CLIP embeddings using HDBSCAN.

This stage answers:

- are these broadly the same type of room or scene?

For example:

- empty-room images may group together
- kitchen images may group together

See [cluster_embeddings](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:707>) and [cluster_same_corner_groups](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:732>).

### 7. Second-stage clustering: same-corner refinement

Inside each semantic cluster, the script clusters again using the combined hybrid feature vector.

This stage is meant to answer:

- are these the same room from the same corner or same viewpoint?

This is where layout and edge features matter more than pure semantics.

### 8. Special refinement for difficult room-view cases

There are two extra refinement steps.

#### A. 4-image pair split

If a semantic cluster contains four highly similar room photos, the script tries to split them into two pairs.

It uses:

- opening profile similarity
- ORB local feature matches

This helps in cases like:

- two photos from one corner
- two photos from another corner

See [maybe_split_quad_cluster](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:336>).

#### B. Broad viewpoint split

If a cluster is still too broad, the script computes a viewpoint similarity matrix using:

- layout similarity
- edge similarity
- opening similarity
- ORB local similarity

Then it uses agglomerative clustering to break a big room cluster into tighter subclusters.

See:

- [viewpoint_similarity_matrix](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:306>)
- [maybe_refine_broad_viewpoint_cluster](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:384>)

The viewpoint score can now be tuned with:

- `--orb-weight`
- `--structure-weight`
- `--local-descriptor-mode`
- `--local-descriptor-weight`
- `--view-linkage`

### 9. Merge-back of over-split subclusters

Sometimes viewpoint splitting is too strict.

To prevent that, the script compares CLIP centroids of the resulting subclusters and merges them back only when they are both semantically close and viewpoint-compatible.

This is controlled by:

- `--semantic-merge-threshold`
- `--merge-view-threshold`

See [merge_semantic_subclusters](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:500>).

### 10. Strict same-corner + same-items mode

If you pass:

- `--strict-same-corner-items`

the second-stage clustering becomes stricter.

In this mode, the script:

- builds CLIP prompt-signature features from a selected prompt set
- requires viewpoint similarity to be high enough
- requires prompt-signature similarity to be high enough
- requires CLIP image embedding similarity to stay above a semantic floor

This is stricter than the default mode because two images are only allowed to stay together if:

- they look like the same corner
- they look like the same visible room/items

Important:

- this is still not full object detection
- it is a stronger CLIP-based proxy for item similarity
- prompt sets are configurable in `configs/prompt_sets.json`
- linkage is configurable with `--strict-linkage`

### 11. Write output

The script creates:

- `output/cluster_0`
- `output/cluster_1`
- `output/noise`
- `output/cluster_0_contact.jpg`
- `output/noise_contact.jpg`

and writes:

- `output/clusters.json`
- `output/match_scores.json`
- `output/run_manifest.json`
- `output/summary.html`

It also records:

- `noise_details` with explicit reasons
- `merge_events` for merge-back decisions
- `skipped_images` for unreadable files

See [copy_clustered_images](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py:914>).

## Output Format

The output JSON looks like this:

```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "images": ["img1.jpg", "img2.jpg"]
    }
  ],
  "noise": ["img10.jpg"]
}
```

`noise` means the image did not belong strongly enough to any cluster under the chosen thresholds.

`match_scores.json` contains pairwise image-to-image matching percentages, sorted from strongest match to weakest match for each image.

## Installation

### Option 1: normal environment install

```powershell
python -m pip install -r requirements.txt
```

### Option 2: keep CLIP vendored locally

If you prefer to keep CLIP inside the project folder:

```powershell
python -m pip install --target .vendor git+https://github.com/openai/CLIP.git
```

The script supports both:

- vendored CLIP in `.vendor`
- normal installed `clip` package in the Python environment

Validate the runtime setup before a full run:

```powershell
python cluster_images.py --validate-setup
```

## Run

Basic run:

```powershell
python cluster_images.py --input ./input --output ./output
```

If the target output folder already exists, the script writes to a timestamped folder automatically.

To overwrite the requested output folder explicitly:

```powershell
python cluster_images.py --input ./input --output ./output --overwrite
```

Force CPU:

```powershell
python cluster_images.py --input ./input --output ./output --device cpu
```

Use a different CLIP model:

```powershell
python cluster_images.py --input ./input --output ./output --model ViT-L/14
```

Use a named preset:

```powershell
python cluster_images.py --input ./input --output ./output --preset strict
```

Use a different prompt set for strict item signatures:

```powershell
python cluster_images.py --input ./input --output ./output --strict-same-corner-items --prompt-set office
```

## Thresholds And What They Do

### HDBSCAN thresholds

- `--preset`
  Applies a named threshold and feature-weight profile before explicit CLI overrides.
  Available presets are `balanced`, `strict`, `loose`, `real_estate`, `office`, and `generic_indoor`.

- `--min-cluster-size`
  Smaller values allow tiny clusters. Larger values force more images to merge or become noise.

- `--min-samples`
  Higher values make clustering stricter and usually produce more noise.

- `--cluster-epsilon`
  Higher values merge nearby groups more easily. Lower values keep groups more separated.

### Feature weights

- `--semantic-weight`
  Increase this to favor CLIP semantics more strongly.

- `--layout-weight`
  Increase this to favor same-room layout and corner similarity.

- `--edge-weight`
  Increase this to favor geometric similarity.

- `--color-weight`
  Increase this if color should matter more.

### Viewpoint refinement thresholds

- `--view-max-cluster-size`
  Optional cap to force large same-room groups to split.

- `--view-similarity-threshold`
  Higher values make same-corner refinement stricter.
  Lower values make viewpoint clusters merge more easily.

- `--semantic-merge-threshold`
  Higher values make merge-back stricter.
  Lower values let semantically similar subclusters merge more easily.

- `--merge-view-threshold`
  Requires minimum viewpoint compatibility before merge-back is allowed.

- `--view-linkage`
  Chooses the broad viewpoint refinement mode: `complete`, `average`, or `graph`.

- `--strict-same-corner-items`
  Turns on the stricter same-corner plus same-items refinement path.

- `--prompt-set`
  Selects the prompt set used for strict item-signature features.

- `--item-similarity-threshold`
  Higher values require stronger room/item similarity.

- `--strict-cluster-threshold`
  Higher values make strict grouping harder.

- `--semantic-similarity-floor`
  Prevents semantically different scenes from merging even if viewpoint looks similar.

- `--strict-linkage`
  Chooses the strict refinement mode: `complete`, `average`, or `graph`.

- `--orb-weight`
  Controls how strongly ORB matching contributes to viewpoint similarity.

- `--structure-weight`
  Controls the additional structural similarity contribution.

- `--local-descriptor-mode`
  Enables the learned local descriptor branch. Current option: `clip_tiles`.

- `--local-descriptor-weight`
  Controls how strongly the learned local descriptor contributes to viewpoint similarity.

### Output controls

- `--overwrite`
  Replace the requested output folder if it already exists.

- `--skip-contact-sheets`
  Skip contact sheet generation for clusters and noise.

- `--skip-html-summary`
  Skip generating `summary.html`.

- `--validate-setup`
  Print dependency and prompt-set validation details, then exit.

## Useful Commands

Balanced default:

```powershell
python cluster_images.py --input ./input --output ./output --preset balanced
```

Stricter same-corner clustering:

```powershell
python cluster_images.py --input ./input --output ./output --preset strict
```

Looser merging:

```powershell
python cluster_images.py --input ./input --output ./output --preset loose
```

Force fewer noise points:

```powershell
python cluster_images.py --input ./input --output ./output --min-cluster-size 3
```

Strict same-corner plus same-items:

```powershell
python cluster_images.py --input ./input --output ./output_strict_items --device cpu --strict-same-corner-items
```

Strict mode with an alternate prompt set and average-link refinement:

```powershell
python cluster_images.py --input ./input --output ./output_strict_items --strict-same-corner-items --prompt-set generic_indoor --strict-linkage average --structure-weight 0.03
```

Strict mode with graph refinement and learned local descriptors:

```powershell
python cluster_images.py --input ./input --output ./output_graph_local --preset strict --strict-same-corner-items --prompt-set generic_indoor --view-linkage graph --strict-linkage graph --local-descriptor-mode clip_tiles --local-descriptor-weight 0.12
```

## Evaluation Harness

Define expected labels in `benchmark/labels_template.json`, or use the real benchmark file `benchmark/real_estate_ground_truth.json`, then compare one or more runs:

```powershell
python evaluate_clusters.py --labels .\benchmark\labels_template.json --clusters .\output\clusters.json .\output_strict\clusters.json --output .\benchmark\comparison_report.json
```

The evaluation report includes:

- pairwise precision
- pairwise recall
- pairwise F1
- cluster purity
- noise rate
- predicted cluster count
- average cluster size

## Current Behavior On Real-Estate Images

With the current implementation, the pipeline is generally good at:

- broad room-type grouping
- splitting one room into multiple corner/view clusters
- keeping unmatched images as noise

It is not yet a full production-grade room-understanding system because it still lacks:

- object detectors
- room-type classifiers
- furniture reasoning
- occupancy modeling

So the current pipeline should be understood as:

- `CLIP semantics`
- plus `grayscale structure`
- plus `threshold-based clustering`

not as:

- full room recognition with detected objects

## Files

- [cluster_images.py](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/cluster_images.py>): main clustering script
- [evaluate_clusters.py](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/evaluate_clusters.py>): benchmark comparison script
- [configs/presets.json](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/configs/presets.json>): preset threshold profiles
- [configs/prompt_sets.json](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/configs/prompt_sets.json>): named strict-mode prompt sets
- [benchmark/README.md](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/benchmark/README.md>): benchmark usage notes
- [benchmark/real_estate_ground_truth.json](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/benchmark/real_estate_ground_truth.json>): real labeled benchmark for the current input set
- [requirements.txt](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/requirements.txt>): Python dependencies
- [input](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/input>): input images
- [output](</c:/Users/Mohammed kaif M/OneDrive/Desktop/clustering images/output>): clustered image output
