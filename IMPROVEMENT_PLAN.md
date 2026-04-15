# Improvement Plan

## Purpose

This document describes the main weaknesses in the current image clustering pipeline, the risks those weaknesses create, and the recommended solutions.

The plan is based on the current implementation in `cluster_images.py` and the supporting project documents.

The current pipeline already has a good structure:

- CLIP for semantic grouping
- handcrafted visual features for viewpoint separation
- HDBSCAN for broad clustering
- Agglomerative clustering for refinement
- JSON outputs for inspection

The main issue is not that the project lacks a clustering pipeline. The main issue is that the current pipeline is still fragile across datasets, thresholds, and runtime environments.

## Implementation Status

Status as of 2026-04-15:

- the primary implementation items from this plan are now present in the codebase
- presets and prompt sets are config-backed
- merge-back now requires semantic and viewpoint compatibility
- noise reassignment, feature caching, setup validation, safer output handling, HTML review output, and benchmark evaluation are implemented
- a learned local descriptor branch and graph-based clustering alternative are implemented
- the benchmark folder now includes real labeled ground truth for the current `input/` dataset

## Goal

Improve clustering quality, robustness, explainability, and operational safety while keeping the system understandable and tunable.

## Success Criteria

The improvement work should achieve the following:

- fewer false merges between different corners or different scenes
- fewer false-noise images
- more stable results across new datasets
- safer and more repeatable runs
- easier debugging and tuning
- clearer outputs for manual review

## Current Pipeline Summary

The current implementation works in this order:

1. discover images
2. load CLIP and create semantic embeddings
3. build layout, edge, and color features
4. optionally build CLIP item-signature features
5. combine features into a hybrid embedding
6. run HDBSCAN on CLIP embeddings for stage-1 semantic grouping
7. refine within each semantic cluster using:
   - local HDBSCAN on hybrid embeddings
   - optional 4-image pair splitting
   - optional agglomerative viewpoint splitting
   - semantic merge-back
8. write output folders, `clusters.json`, and `match_scores.json`

This is a strong baseline, but several parts are highly threshold-sensitive and dataset-dependent.

## Executive Summary

The most important current problems are:

1. thresholds are too dataset-sensitive
2. semantic merge-back can undo correct viewpoint splits
3. strict mode can create too much noise
4. local viewpoint matching is brittle in weak-texture images
5. repeated feature computation increases runtime unnecessarily
6. environment setup is not fully reliable across machines
7. outputs are useful for debugging but weak for large-scale QA
8. there is no formal evaluation harness

The highest-value near-term fixes are:

1. add named threshold presets
2. tighten or redesign semantic merge-back logic
3. add a noise reassignment pass
4. cache per-image derived features
5. add contact sheets or HTML summaries
6. add a benchmark and metric-based evaluation script

## Primary Problems, Risks, and Solutions

## 1. Threshold Sensitivity Across Datasets

### Problem

The current pipeline relies on fixed thresholds for:

- HDBSCAN behavior
- viewpoint refinement
- semantic merge-back
- strict-mode gating

Examples in the current code include:

- `view_similarity_threshold`
- `semantic_merge_threshold`
- `item_similarity_threshold`
- `strict_cluster_threshold`
- `semantic_similarity_floor`

### Why it matters

Thresholds that work for one dataset can fail badly on another.

Examples:

- real-estate interiors may cluster reasonably
- office scenes or detail-heavy datasets may over-split
- more diverse images may produce too much noise
- very similar images may get over-merged

### Risk

- high risk of unstable clustering across folders
- high manual tuning cost
- low confidence when moving to new domains

### Root cause

The implementation uses fixed thresholds but does not have:

- named profiles
- dataset-specific presets
- auto-tuning
- benchmark-guided selection

### Recommended solution

Introduce a configuration layer with named presets.

Minimum preset set:

- `balanced`
- `strict`
- `loose`

Optional domain presets:

- `real_estate`
- `office`
- `generic_indoor`

### Implementation steps

1. create a config object or config file for threshold bundles
2. add `--preset` as a CLI argument
3. keep existing flags as overrides on top of the preset
4. log the final resolved settings at runtime

### Expected impact

- faster reuse across datasets
- fewer ad hoc experiments
- easier debugging because the full threshold profile is explicit

## 2. Semantic Merge-Back Can Undo Correct Viewpoint Splits

### Problem

The current default pipeline may:

1. correctly split a broad room cluster into viewpoint subclusters
2. then merge those subclusters back together if their CLIP centroids are too similar

This behavior is implemented in `merge_semantic_subclusters()`.

### Why it matters

This is one of the most important current failure modes.

A cluster can look correct from a viewpoint perspective, but the semantic merge step may still collapse it into one larger cluster because CLIP sees the groups as semantically similar.

### Risk

- high risk of false merges
- same-room but different-corner views collapse into one cluster
- users lose the same-corner structure they expect

### Root cause

Merge-back currently depends on semantic centroid similarity only.

That means:

- semantic agreement is rewarded
- viewpoint separation is not preserved strongly enough during merge-back

### Recommended solution

Redesign merge-back so it requires both:

- semantic compatibility
- viewpoint compatibility

Do not merge only because CLIP centroids are close.

### Implementation options

Option A:

- raise `semantic_merge_threshold`
- this is a fast mitigation, not a full fix

Option B:

- merge only if semantic similarity is high and average viewpoint compatibility is above a second threshold

Option C:

- disable merge-back for subclusters produced by strong viewpoint evidence

### Implementation steps

1. compute cluster-to-cluster average viewpoint compatibility
2. require a semantic threshold and a viewpoint threshold
3. log why a merge happened
4. include merge reasons in diagnostics

### Expected impact

- fewer false merges
- more reliable same-corner grouping
- better alignment with what users see in the output folders

## 3. Strict Mode Produces Too Much Noise

### Problem

The strict path uses hard gates:

- viewpoint must pass
- item similarity must pass
- semantic similarity must pass

Then it uses complete-link agglomerative clustering, which is sensitive to weakest-link pairs.

### Why it matters

This is good for clean clusters, but it often rejects valid images such as:

- detail shots
- partial views
- low-texture images
- lighting-shifted variants

### Risk

- high risk of false noise
- users may think the model failed even when images are related
- important images may disappear into `noise/`

### Root cause

- hard binary gates
- complete-link sensitivity
- no repair step after clustering

### Recommended solution

Add a post-clustering reassignment pass for noise images.

### Reassignment rule

A noise image should be reassigned only if:

- semantic similarity to a cluster is above a guardrail
- viewpoint compatibility with the cluster is above a guardrail
- item similarity is above a guardrail in strict mode
- average compatibility to top cluster members is strong enough

### Additional improvement

Test alternatives to strict complete-link clustering:

- average-link agglomerative
- graph-based clustering
- thresholded connected components with validation rules

### Implementation steps

1. implement noise reassignment after primary clustering
2. compare complete-link vs average-link on benchmark cases
3. log reassignment reasons in `match_scores.json`

### Expected impact

- fewer false-noise images
- better handling of detail views
- more usable strict mode

## 4. ORB and Handcrafted Local Matching Are Brittle

### Problem

The current viewpoint similarity includes ORB-based local matching and opening-profile heuristics.

These are useful, but brittle when:

- surfaces are low-texture
- exposure changes significantly
- blur is present
- the scene is simple or repetitive

### Why it matters

If local viewpoint features are weak, the refinement stage becomes unstable.

### Risk

- medium risk of incorrect viewpoint similarity
- cluster splits may be inconsistent
- one dataset may behave very differently from another

### Root cause

ORB depends on detectable local keypoints and stable low-level structure.

### Recommended solution

Keep ORB as a supporting signal, not the main decision maker.

Evaluate adding one stronger structural signal:

- SSIM-like structural scoring
- DINO or DINOv2 features
- learned local descriptors
- image patch embeddings

### Implementation steps

1. make ORB weight explicitly configurable if it is not already intended to be fixed
2. add one alternative structural feature behind a flag
3. compare its effect on low-texture and exposure-shifted data

### Expected impact

- more stable viewpoint scores
- less dependency on texture-heavy scenes

## 5. Repeated Image Loading and Feature Recalculation Increase Runtime

### Problem

The current code reloads images and recomputes viewpoint-derived features in multiple places.

Examples:

- viewpoint refinement
- quad-cluster splitting
- match-score generation

### Why it matters

The system becomes slower than necessary and harder to profile.

### Risk

- medium risk of poor runtime on larger datasets
- duplicated work makes future complexity more expensive
- repeated image reads increase I/O cost

### Root cause

Feature computation is functionally clean, but not centralized in a reusable cache.

### Recommended solution

Introduce a per-image feature cache.

Cache once:

- RGB image if safe to keep
- layout vector
- edge vector
- opening profile
- ORB descriptors

### Implementation steps

1. create a feature-cache object keyed by image path
2. compute image-derived features once
3. pass cached features into refinement and scoring functions
4. optionally add an on-disk cache later if dataset size grows

### Expected impact

- lower runtime
- easier profiling
- simpler future experimentation

## 6. Environment and Dependency Setup Are Fragile

### Problem

The runtime environment has already shown setup fragility:

- `cv2` was used in code but not originally listed in `requirements.txt`
- vendored CLIP access could fail depending on file permissions
- HDBSCAN parallel execution caused a Windows permission issue in this environment

### Why it matters

Even good clustering code loses value if the setup is fragile or machine-specific.

### Risk

- high risk of onboarding friction
- scripts may fail differently across machines
- reproducibility is reduced

### Root cause

- environment assumptions are not fully standardized
- local vendor usage and normal package usage can diverge
- no install validation or smoke test step exists

### Recommended solution

Harden the setup workflow.

### Implementation steps

1. keep `requirements.txt` aligned with actual imports
2. add a `python cluster_images.py --help` smoke test to setup docs
3. add a tiny validation section to the README
4. add a startup self-check for critical dependencies
5. consider removing or simplifying vendor-first behavior if it remains unreliable

### Expected impact

- easier setup
- fewer environment-specific failures
- more predictable execution

## 7. Output Handling Is Not Safe Enough for Iterative Experiments

### Problem

The script recreates the output folder before writing new results.

That makes experiments easy to overwrite.

### Why it matters

Users often compare multiple runs with different thresholds. If the output directory is destructive, results can be lost accidentally.

### Risk

- medium risk of accidental result loss
- weaker experiment traceability
- harder threshold comparison

### Root cause

The current workflow assumes a single target output folder per run.

### Recommended solution

Add safer output semantics.

### Implementation steps

1. add `--overwrite` as an explicit flag
2. if overwrite is not set, create a timestamped output directory
3. store the resolved run settings in the output folder
4. optionally add a run manifest JSON

### Expected impact

- safer experimentation
- easier result comparison
- better reproducibility

## 8. Output Review Is Too Manual for Large Runs

### Problem

`clusters.json` and `match_scores.json` are useful, but they are still slow for humans to review at scale.

### Why it matters

Cluster quality tuning is visual. If manual review is slow, tuning becomes slow.

### Risk

- medium risk of slow iteration
- threshold changes become hard to compare
- debugging false merges or false noise takes too long

### Root cause

The output is machine-readable first, human-review-friendly second.

### Recommended solution

Add visual review artifacts.

### Implementation steps

1. generate contact sheets per cluster
2. optionally generate an HTML summary page
3. include representative image per cluster
4. include top near-miss images
5. include run settings in the summary

### Expected impact

- much faster QA
- easier cluster comparison
- better debugging of borderline cases

## 9. There Is No Formal Evaluation Harness

### Problem

Threshold tuning is currently manual and visual.

There is no benchmark set, no quality metric dashboard, and no script for comparing runs objectively.

### Why it matters

Without metrics, it is hard to know whether a change truly improved the system.

### Risk

- high risk of subjective tuning
- regressions may go unnoticed
- engineering time may be spent on changes that do not help

### Root cause

The project has diagnostics, but not formal evaluation.

### Recommended solution

Create a lightweight benchmark harness.

### Suggested metrics

- cluster purity
- pairwise precision
- pairwise recall
- noise rate
- number of clusters
- average cluster size

### Implementation steps

1. create a small labeled benchmark dataset
2. define expected group labels
3. build an evaluation script that runs multiple profiles
4. export a summary report for comparison

### Expected impact

- more reliable tuning
- easier regression detection
- clearer decision-making for algorithm changes

## 10. Prompt-Signature Item Features Are Domain-Biased

### Problem

`STRICT_ITEM_PROMPTS` is fixed and currently biased toward indoor real-estate style scenes.

### Why it matters

The strict item feature may work better for one domain than another.

### Risk

- medium risk of poor strict-mode performance outside the original domain
- office, retail, or mixed indoor scenes may be described poorly

### Root cause

Prompt coverage is narrow and static.

### Recommended solution

Make prompt sets configurable by domain.

### Implementation steps

1. move prompt sets into a separate config file
2. support multiple named prompt sets
3. add a generic default prompt set
4. select prompt set with a CLI flag

### Expected impact

- better strict-mode portability
- easier adaptation to new domains

## Risk Register

| Risk | Likelihood | Impact | Priority | Recommended mitigation |
|---|---|---:|---:|---|
| Thresholds fail on new datasets | High | High | P0 | Add presets and benchmark evaluation |
| Semantic merge-back over-merges distinct viewpoints | High | High | P0 | Redesign merge-back to include viewpoint compatibility |
| Strict mode produces too much noise | High | High | P0 | Add noise reassignment and evaluate alternative linkage |
| Local ORB/viewpoint signals fail on weak-texture scenes | Medium | Medium | P1 | Add alternative structural signal and tune weights |
| Runtime grows due to repeated feature extraction | Medium | Medium | P1 | Add per-image caching |
| Environment setup breaks across machines | High | Medium | P0 | Harden dependencies and add setup validation |
| Output folders are overwritten accidentally | Medium | Medium | P1 | Add explicit overwrite control and timestamped outputs |
| Human review remains slow | Medium | Medium | P1 | Add contact sheets or HTML report |
| Regressions go unnoticed | High | High | P0 | Build evaluation harness |
| Prompt set performs poorly on new domains | Medium | Medium | P2 | Make prompts configurable |

## Prioritized Delivery Plan

## Phase 0: Stability and Safety

Primary focus:

- make the system safer to run
- reduce environment friction
- improve reproducibility

Tasks:

1. keep dependencies aligned with imports
2. add setup smoke-test instructions
3. add timestamped outputs or `--overwrite`
4. log resolved threshold settings

## Phase 1: Clustering Quality Improvements

Primary focus:

- reduce false merges
- reduce false noise
- make thresholds easier to manage

Tasks:

1. add named presets
2. redesign semantic merge-back with viewpoint compatibility
3. add post-pass noise reassignment
4. test stricter and safer merge logic on known failure cases

## Phase 2: Performance and Reviewability

Primary focus:

- improve runtime
- improve cluster review workflow

Tasks:

1. add feature caching
2. generate contact sheets
3. optionally generate HTML summaries
4. include run metadata in outputs

## Phase 3: Evaluation and Model Upgrades

Primary focus:

- make improvements measurable
- test stronger alternatives

Tasks:

1. create a benchmark set
2. add evaluation metrics and comparison script
3. test average-link or graph-based strict clustering
4. test alternative structural features beyond ORB
5. add domain-configurable prompt sets

## Recommended Immediate Next Steps

If the goal is to improve quality quickly without a major rewrite, the best next three changes are:

1. add threshold presets
2. redesign merge-back or raise its safety conditions
3. add a noise reassignment post-pass

If the goal is to improve operator workflow quickly, the best next three changes are:

1. add timestamped output folders
2. add contact sheets
3. log full resolved run settings

If the goal is to improve long-term engineering quality, the best next three changes are:

1. add a benchmark harness
2. add feature caching
3. add environment validation

## Example Near-Term Deliverables

### Deliverable A: Preset system

Output:

- `--preset balanced|strict|loose`
- config-backed thresholds
- resolved settings written to logs and output manifest

### Deliverable B: Safer merge-back

Output:

- merge-back checks semantic and viewpoint compatibility
- merge events recorded with reasons
- fewer false merges in same-room multi-corner cases

### Deliverable C: Noise reassignment

Output:

- noise images evaluated against nearby clusters
- reassignment only when guardrails pass
- fewer false-noise detail images

### Deliverable D: Evaluation harness

Output:

- benchmark folder
- label file
- comparison script
- report with pairwise precision, recall, purity, and noise rate

## Conclusion

The current project is a solid baseline, but the main risks are not in the existence of the clustering pipeline itself. The main risks are:

- threshold fragility
- false merge behavior after correct splits
- strict-mode noise inflation
- limited evaluation and review tooling
- environment and workflow instability

The best improvement strategy is:

1. stabilize the runtime and experiment workflow
2. fix the merge-back and noise-handling weaknesses
3. add evaluation and review tools
4. then explore heavier algorithm upgrades

That sequence gives the best balance of engineering cost, clustering quality, and practical usability.
