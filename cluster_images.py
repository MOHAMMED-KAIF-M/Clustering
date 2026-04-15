from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from html import escape
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"

import numpy as np
import torch
import cv2
from PIL import Image, ImageFilter, ImageOps
from sklearn.cluster import AgglomerativeClustering, HDBSCAN


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ORB_EXTRACTOR = cv2.ORB_create(1200)
ORB_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
PRESET_CONFIG_PATH = CONFIG_DIR / "presets.json"
PROMPT_SET_CONFIG_PATH = CONFIG_DIR / "prompt_sets.json"
DEFAULT_PROMPT_SET_CONFIGS: dict[str, list[str]] = {
    "real_estate": [
        "a real estate interior photo of an empty room",
        "a real estate interior photo of a furnished room",
        "a real estate interior photo of a bedroom",
        "a real estate interior photo of a living room",
        "a real estate interior photo of a hall",
        "a real estate interior photo of a kitchen",
        "a real estate interior photo of a bathroom",
        "a real estate interior photo of a balcony",
        "a bed",
        "a bedside table",
        "a sofa",
        "a television",
        "a dining table",
        "chairs",
        "a kitchen island",
        "kitchen cabinets",
        "a countertop",
        "a stove",
        "a sink",
        "a refrigerator",
        "a wardrobe",
        "a toilet",
        "a bathtub",
        "a shower",
        "a window",
        "a sliding glass door",
    ],
    "office": [
        "an office room",
        "a meeting room",
        "a conference table",
        "an office desk",
        "a desktop computer",
        "a laptop",
        "office chairs",
        "a whiteboard",
        "a glass partition",
        "a cubicle",
        "a reception desk",
        "a cabinet",
        "a printer",
        "a monitor",
        "an indoor hallway",
        "a window",
        "a door",
        "an office lounge",
    ],
    "generic_indoor": [
        "an indoor room",
        "an interior scene",
        "a living space",
        "furniture",
        "a table",
        "chairs",
        "a sofa",
        "a bed",
        "a desk",
        "a cabinet",
        "a window",
        "a door",
        "a hallway",
        "a kitchen",
        "a bathroom",
        "a bedroom",
        "a wall",
        "a floor",
        "ceiling lights",
    ],
}
DEFAULT_PRESET_CONFIGS: dict[str, dict[str, float]] = {
    "balanced": {
        "min_cluster_size": 2,
        "min_samples": 1,
        "cluster_epsilon": 0.0,
        "semantic_weight": 0.45,
        "layout_weight": 0.35,
        "edge_weight": 0.15,
        "color_weight": 0.05,
        "view_similarity_threshold": 0.34,
        "semantic_merge_threshold": 0.95,
        "item_similarity_threshold": 0.84,
        "strict_cluster_threshold": 0.56,
        "semantic_similarity_floor": 0.90,
        "orb_weight": 0.15,
        "structure_weight": 0.0,
        "local_descriptor_weight": 0.0,
    },
    "strict": {
        "min_cluster_size": 2,
        "min_samples": 2,
        "cluster_epsilon": 0.0,
        "semantic_weight": 0.40,
        "layout_weight": 0.35,
        "edge_weight": 0.20,
        "color_weight": 0.05,
        "view_similarity_threshold": 0.38,
        "semantic_merge_threshold": 0.98,
        "item_similarity_threshold": 0.86,
        "strict_cluster_threshold": 0.60,
        "semantic_similarity_floor": 0.92,
        "orb_weight": 0.12,
        "structure_weight": 0.03,
        "local_descriptor_weight": 0.0,
    },
    "loose": {
        "min_cluster_size": 2,
        "min_samples": 1,
        "cluster_epsilon": 0.08,
        "semantic_weight": 0.50,
        "layout_weight": 0.30,
        "edge_weight": 0.15,
        "color_weight": 0.05,
        "view_similarity_threshold": 0.28,
        "semantic_merge_threshold": 0.93,
        "item_similarity_threshold": 0.80,
        "strict_cluster_threshold": 0.52,
        "semantic_similarity_floor": 0.86,
        "orb_weight": 0.12,
        "structure_weight": 0.03,
        "local_descriptor_weight": 0.0,
    },
    "real_estate": {
        "min_cluster_size": 2,
        "min_samples": 1,
        "cluster_epsilon": 0.0,
        "semantic_weight": 0.45,
        "layout_weight": 0.35,
        "edge_weight": 0.15,
        "color_weight": 0.05,
        "view_similarity_threshold": 0.34,
        "semantic_merge_threshold": 0.97,
        "item_similarity_threshold": 0.84,
        "strict_cluster_threshold": 0.56,
        "semantic_similarity_floor": 0.90,
        "orb_weight": 0.15,
        "structure_weight": 0.0,
        "local_descriptor_weight": 0.0,
    },
    "office": {
        "min_cluster_size": 2,
        "min_samples": 2,
        "cluster_epsilon": 0.04,
        "semantic_weight": 0.42,
        "layout_weight": 0.30,
        "edge_weight": 0.18,
        "color_weight": 0.10,
        "view_similarity_threshold": 0.30,
        "semantic_merge_threshold": 0.96,
        "item_similarity_threshold": 0.82,
        "strict_cluster_threshold": 0.55,
        "semantic_similarity_floor": 0.88,
        "orb_weight": 0.10,
        "structure_weight": 0.05,
        "local_descriptor_weight": 0.0,
    },
    "generic_indoor": {
        "min_cluster_size": 2,
        "min_samples": 1,
        "cluster_epsilon": 0.03,
        "semantic_weight": 0.46,
        "layout_weight": 0.30,
        "edge_weight": 0.14,
        "color_weight": 0.10,
        "view_similarity_threshold": 0.30,
        "semantic_merge_threshold": 0.95,
        "item_similarity_threshold": 0.80,
        "strict_cluster_threshold": 0.54,
        "semantic_similarity_floor": 0.87,
        "orb_weight": 0.12,
        "structure_weight": 0.04,
        "local_descriptor_weight": 0.0,
    },
}


def load_json_config(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default
    return payload if isinstance(payload, dict) else default


PROMPT_SET_CONFIGS = load_json_config(PROMPT_SET_CONFIG_PATH, DEFAULT_PROMPT_SET_CONFIGS)
PRESET_CONFIGS = load_json_config(PRESET_CONFIG_PATH, DEFAULT_PRESET_CONFIGS)


@dataclass
class CachedImageFeatures:
    path: Path
    image: Image.Image
    layout: np.ndarray
    edge: np.ndarray
    color: np.ndarray
    opening: np.ndarray
    structure: np.ndarray
    orb_descriptors: np.ndarray | None
    learned_local_descriptor: np.ndarray | None = None


def load_clip_module() -> tuple[object, str]:
    clip_init = VENDOR_DIR / "clip" / "__init__.py"
    if clip_init.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                "clip",
                clip_init,
                submodule_search_locations=[str(clip_init.parent)],
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load CLIP package from {clip_init}")

            clip = importlib.util.module_from_spec(spec)
            sys.modules["clip"] = clip
            spec.loader.exec_module(clip)
            if not hasattr(clip, "load"):
                raise RuntimeError(f"Loaded CLIP package from {clip_init}, but clip.load is missing.")
            return clip, str(clip_init)
        except (OSError, PermissionError):
            sys.modules.pop("clip", None)

    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "CLIP is not installed. Install dependencies from requirements.txt or install it into .vendor."
        ) from exc

    if not hasattr(clip, "load"):
        raise RuntimeError("Imported clip module does not expose clip.load. Check your CLIP installation.")
    return clip, getattr(clip, "__file__", "installed package")


def load_clip_runtime(
    model_name: str,
    device: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> tuple[object, object, object]:
    clip, clip_source = load_clip_module()
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded CLIP from %s", clip_source)
    model, preprocess = clip.load(model_name, device=device, jit=False, download_root=str(cache_dir))
    model.eval()
    return clip, model, preprocess


def cli_flag_was_provided(raw_args: list[str], argument_name: str) -> bool:
    flag = f"--{argument_name.replace('_', '-')}"
    return any(raw_arg == flag or raw_arg.startswith(f"{flag}=") for raw_arg in raw_args)


def apply_preset_defaults(args: argparse.Namespace, raw_args: list[str]) -> argparse.Namespace:
    preset_values = PRESET_CONFIGS[args.preset]
    for argument_name, value in preset_values.items():
        if not cli_flag_was_provided(raw_args, argument_name):
            setattr(args, argument_name, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster images from an input folder using CLIP + HDBSCAN.")
    parser.add_argument("--input", default="input", help="Input image folder.")
    parser.add_argument("--output", default="output", help="Output folder for clustered images.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS.keys()),
        default="balanced",
        help="Apply a named threshold/weight preset before CLI overrides.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=sorted(PROMPT_SET_CONFIGS.keys()),
        default="real_estate",
        help="Prompt set used for strict same-corner+items CLIP item signatures.",
    )
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="HDBSCAN min_cluster_size.")
    parser.add_argument("--min-samples", type=int, default=1, help="HDBSCAN min_samples.")
    parser.add_argument(
        "--cluster-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon. Higher values merge nearby clusters more aggressively.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.45,
        help="Weight for CLIP semantic features.",
    )
    parser.add_argument(
        "--layout-weight",
        type=float,
        default=0.35,
        help="Weight for grayscale layout features. Increase this to favor same-corner clustering.",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.15,
        help="Weight for edge-map features. Increase this to favor similar geometry and room structure.",
    )
    parser.add_argument(
        "--color-weight",
        type=float,
        default=0.05,
        help="Weight for color histogram features.",
    )
    parser.add_argument(
        "--view-max-cluster-size",
        type=int,
        default=None,
        help="Optional cap for second-stage same-corner clusters. Lower values force large room groups to split.",
    )
    parser.add_argument(
        "--view-similarity-threshold",
        type=float,
        default=0.34,
        help="Minimum pairwise viewpoint similarity used when refining broad same-room clusters into tighter same-corner groups.",
    )
    parser.add_argument(
        "--semantic-merge-threshold",
        type=float,
        default=0.95,
        help="Merge back same-room subclusters when their CLIP centroid similarity is above this threshold.",
    )
    parser.add_argument(
        "--merge-view-threshold",
        type=float,
        default=0.28,
        help="Minimum average viewpoint compatibility required before semantically similar subclusters can merge back.",
    )
    parser.add_argument(
        "--strict-same-corner-items",
        action="store_true",
        help="Use stricter second-stage clustering that requires both close viewpoint similarity and close item similarity.",
    )
    parser.add_argument(
        "--item-similarity-threshold",
        type=float,
        default=0.84,
        help="Minimum CLIP prompt-signature similarity required for images to remain in the same strict cluster.",
    )
    parser.add_argument(
        "--strict-cluster-threshold",
        type=float,
        default=0.56,
        help="Minimum combined strict similarity used by complete-link clustering in strict same-corner+items mode.",
    )
    parser.add_argument(
        "--semantic-similarity-floor",
        type=float,
        default=0.90,
        help="Minimum CLIP image embedding similarity required before two images can be grouped in strict mode.",
    )
    parser.add_argument(
        "--view-linkage",
        choices=["complete", "average", "graph"],
        default="complete",
        help="Clustering mode used for broad viewpoint refinement splits.",
    )
    parser.add_argument(
        "--strict-linkage",
        choices=["complete", "average", "graph"],
        default="complete",
        help="Clustering mode used in strict same-corner+items refinement.",
    )
    parser.add_argument(
        "--orb-weight",
        type=float,
        default=0.15,
        help="Relative weight of ORB local matching inside viewpoint similarity.",
    )
    parser.add_argument(
        "--structure-weight",
        type=float,
        default=0.0,
        help="Relative weight of the additional structural similarity signal inside viewpoint similarity.",
    )
    parser.add_argument(
        "--local-descriptor-mode",
        choices=["none", "clip_tiles"],
        default="none",
        help="Optional learned local descriptor used inside viewpoint similarity.",
    )
    parser.add_argument(
        "--local-descriptor-weight",
        type=float,
        default=0.0,
        help="Relative weight of the learned local descriptor inside viewpoint similarity.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output folder if it already exists. Otherwise a timestamped folder is created.",
    )
    parser.add_argument(
        "--skip-contact-sheets",
        action="store_true",
        help="Skip generating contact sheets for clusters and noise.",
    )
    parser.add_argument(
        "--skip-html-summary",
        action="store_true",
        help="Skip HTML summary generation.",
    )
    parser.add_argument(
        "--validate-setup",
        action="store_true",
        help="Validate dependencies and configuration, print the result, and exit.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device.")
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("cluster_images")


def resolve_device(raw_device: str) -> str:
    if raw_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return raw_device


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb_image(image_path: Path) -> Image.Image | None:
    try:
        with Image.open(image_path) as image:
            normalized = ImageOps.exif_transpose(image)
            return normalized.convert("RGB")
    except (OSError, ValueError):
        return None


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def image_to_layout_vector(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    grayscale = image.resize(size, Image.Resampling.BICUBIC).convert("L")
    vector = np.asarray(grayscale, dtype=np.float32).reshape(-1)
    vector -= vector.mean()
    return l2_normalize(vector.reshape(1, -1))[0]


def image_to_edge_vector(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    edge_image = image.resize(size, Image.Resampling.BICUBIC).convert("L").filter(ImageFilter.FIND_EDGES)
    vector = np.asarray(edge_image, dtype=np.float32).reshape(-1)
    vector -= vector.mean()
    return l2_normalize(vector.reshape(1, -1))[0]


def image_to_color_histogram(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV").resize((48, 48), Image.Resampling.BICUBIC), dtype=np.float32)
    histograms: list[np.ndarray] = []

    channel_bins = [(0.0, 255.0, 12), (0.0, 255.0, 6), (0.0, 255.0, 6)]
    for channel_index, (low, high, bins) in enumerate(channel_bins):
        channel = hsv[:, :, channel_index].reshape(-1)
        histogram, _ = np.histogram(channel, bins=bins, range=(low, high), density=False)
        histograms.append(histogram.astype(np.float32))

    return l2_normalize(np.concatenate(histograms).reshape(1, -1))[0]


def image_to_opening_profile(image: Image.Image) -> np.ndarray:
    resized = image.resize((64, 64), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]

    bright_mask = array.mean(axis=2) > 210.0
    blue_mask = (blue > green + 8.0) & (blue > red + 8.0) & (blue > 120.0)
    opening_mask = (bright_mask | blue_mask).astype(np.float32)

    y_coords, _ = np.mgrid[0:64, 0:64]
    weight = np.ones_like(opening_mask, dtype=np.float32)
    weight[y_coords < 40] += 0.8
    weight[y_coords < 24] += 0.5
    weighted_mask = opening_mask * weight

    feature = np.concatenate(
        [
            weighted_mask.mean(axis=0),
            weighted_mask.mean(axis=1),
            np.array(
                [
                    weighted_mask[:, :21].mean(),
                    weighted_mask[:, 21:43].mean(),
                    weighted_mask[:, 43:].mean(),
                ],
                dtype=np.float32,
            ),
        ]
    )
    feature -= feature.mean()
    return l2_normalize(feature.reshape(1, -1))[0]


def image_to_structure_vector(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    grayscale = np.asarray(image.resize(size, Image.Resampling.BICUBIC).convert("L"), dtype=np.float32)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    vertical = np.abs(np.diff(blurred, axis=0, prepend=blurred[:1, :]))
    horizontal = np.abs(np.diff(blurred, axis=1, prepend=blurred[:, :1]))
    structure = np.concatenate(
        [
            blurred.reshape(-1),
            vertical.reshape(-1),
            horizontal.reshape(-1),
        ]
    ).astype(np.float32, copy=False)
    structure -= structure.mean()
    return l2_normalize(structure.reshape(1, -1))[0]


def image_to_orb_descriptors(image: Image.Image) -> np.ndarray | None:
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, descriptors = ORB_EXTRACTOR.detectAndCompute(gray, None)
    return descriptors


def orb_similarity_score(descriptors_a: np.ndarray | None, descriptors_b: np.ndarray | None) -> float:
    if descriptors_a is None or descriptors_b is None or not len(descriptors_a) or not len(descriptors_b):
        return 0.0

    matches = ORB_MATCHER.match(descriptors_a, descriptors_b)
    if not matches:
        return 0.0

    good_matches = [match for match in matches if match.distance < 42]
    denominator = max(min(len(descriptors_a), len(descriptors_b)), 1)
    return float(len(good_matches) / denominator)


def structural_similarity_score(structure_a: np.ndarray, structure_b: np.ndarray) -> float:
    return float(np.clip(np.dot(structure_a, structure_b), 0.0, 1.0))


def generate_clip_tile_crops(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    crop_width = max(64, int(width * 0.72))
    crop_height = max(64, int(height * 0.72))
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    candidate_boxes = [
        (0, 0, crop_width, crop_height),
        (width - crop_width, 0, width, crop_height),
        (0, height - crop_height, crop_width, height),
        (width - crop_width, height - crop_height, width, height),
        (
            max(0, (width - crop_width) // 2),
            max(0, (height - crop_height) // 2),
            max(0, (width - crop_width) // 2) + crop_width,
            max(0, (height - crop_height) // 2) + crop_height,
        ),
    ]

    crops: list[Image.Image] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for left, top, right, bottom in candidate_boxes:
        box = (
            max(0, min(left, width - crop_width)),
            max(0, min(top, height - crop_height)),
            max(0, min(left, width - crop_width)) + crop_width,
            max(0, min(top, height - crop_height)) + crop_height,
        )
        if box in seen_boxes:
            continue
        seen_boxes.add(box)
        crops.append(image.crop(box))
    return crops


def learned_local_similarity_score(
    descriptor_a: np.ndarray | None,
    descriptor_b: np.ndarray | None,
) -> float:
    if descriptor_a is None or descriptor_b is None or not descriptor_a.size or not descriptor_b.size:
        return 0.0

    similarity = np.clip(descriptor_a @ descriptor_b.T, -1.0, 1.0).astype(np.float32, copy=False)
    forward = float(similarity.max(axis=1).mean())
    backward = float(similarity.max(axis=0).mean())
    return float(np.clip(0.5 * (forward + backward), 0.0, 1.0))


def populate_clip_tile_descriptors(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    feature_cache: dict[Path, CachedImageFeatures],
    logger: logging.Logger,
) -> None:
    pending_paths = [image_path for image_path in image_paths if feature_cache[image_path].learned_local_descriptor is None]
    if not pending_paths:
        return

    _, model, preprocess = load_clip_runtime(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        logger=logger,
    )

    with torch.no_grad():
        for index, image_path in enumerate(pending_paths, start=1):
            record = feature_cache[image_path]
            crops = generate_clip_tile_crops(record.image)
            crop_tensor = torch.stack([preprocess(crop) for crop in crops]).to(device)
            crop_features = model.encode_image(crop_tensor).float()
            crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
            record.learned_local_descriptor = crop_features.cpu().numpy().astype(np.float32, copy=False)
            if index % 8 == 0 or index == len(pending_paths):
                logger.info(
                    "Built learned local descriptors for %s/%s images",
                    index,
                    len(pending_paths),
                )


def graph_connected_components(similarity: np.ndarray, threshold: float, min_cluster_size: int) -> tuple[list[list[int]], list[int]]:
    size = similarity.shape[0]
    if size == 0:
        return [], []

    adjacency = similarity >= threshold
    np.fill_diagonal(adjacency, True)
    visited = np.zeros(size, dtype=bool)
    clusters: list[list[int]] = []
    noise_indices: list[int] = []

    for start_index in range(size):
        if visited[start_index]:
            continue
        stack = [start_index]
        visited[start_index] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = np.where(adjacency[current])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        component.sort()
        if len(component) >= min_cluster_size:
            clusters.append(component)
        else:
            noise_indices.extend(component)

    return clusters, sorted(noise_indices)


def build_feature_cache(
    image_paths: list[Path],
    logger: logging.Logger,
) -> tuple[dict[Path, CachedImageFeatures], list[Path], list[str]]:
    feature_cache: dict[Path, CachedImageFeatures] = {}
    valid_paths: list[Path] = []
    skipped_images: list[str] = []

    for image_path in image_paths:
        image = load_rgb_image(image_path)
        if image is None:
            logger.warning("Skipping unreadable image during feature cache build: %s", image_path)
            skipped_images.append(image_path.name)
            continue

        feature_cache[image_path] = CachedImageFeatures(
            path=image_path,
            image=image,
            layout=image_to_layout_vector(image, size=(24, 24)),
            edge=image_to_edge_vector(image, size=(24, 24)),
            color=image_to_color_histogram(image),
            opening=image_to_opening_profile(image),
            structure=image_to_structure_vector(image, size=(24, 24)),
            orb_descriptors=image_to_orb_descriptors(image),
            learned_local_descriptor=None,
        )
        valid_paths.append(image_path)

    return feature_cache, valid_paths, skipped_images


def viewpoint_similarity_matrix(
    image_paths: list[Path],
    logger: logging.Logger,
    feature_cache: dict[Path, CachedImageFeatures] | None = None,
    orb_weight: float = 0.15,
    structure_weight: float = 0.0,
    local_descriptor_weight: float = 0.0,
) -> np.ndarray | None:
    records: list[CachedImageFeatures] = []
    for image_path in image_paths:
        record = feature_cache.get(image_path) if feature_cache is not None else None
        if record is None:
            image = load_rgb_image(image_path)
        else:
            image = record.image
        if image is None:
            logger.warning("Skipping unreadable image during viewpoint refinement: %s", image_path)
            return None
        if record is None:
            record = CachedImageFeatures(
                path=image_path,
                image=image,
                layout=image_to_layout_vector(image, size=(24, 24)),
                edge=image_to_edge_vector(image, size=(24, 24)),
                color=image_to_color_histogram(image),
                opening=image_to_opening_profile(image),
                structure=image_to_structure_vector(image, size=(24, 24)),
                orb_descriptors=image_to_orb_descriptors(image),
                learned_local_descriptor=None,
            )
        records.append(record)

    raw_weights = np.array(
        [
            0.35,
            0.25,
            0.25,
            max(0.0, orb_weight),
            max(0.0, structure_weight),
            max(0.0, local_descriptor_weight),
        ],
        dtype=np.float32,
    )
    weights = raw_weights / raw_weights.sum()

    size = len(image_paths)
    similarity = np.eye(size, dtype=np.float32)
    for first_index in range(size):
        for second_index in range(first_index + 1, size):
            score = (
                weights[0] * float(np.dot(records[first_index].layout, records[second_index].layout))
                + weights[1] * float(np.dot(records[first_index].edge, records[second_index].edge))
                + weights[2] * float(np.dot(records[first_index].opening, records[second_index].opening))
                + weights[3] * orb_similarity_score(
                    records[first_index].orb_descriptors,
                    records[second_index].orb_descriptors,
                )
                + weights[4] * structural_similarity_score(records[first_index].structure, records[second_index].structure)
                + weights[5] * learned_local_similarity_score(
                    records[first_index].learned_local_descriptor,
                    records[second_index].learned_local_descriptor,
                )
            )
            similarity[first_index, second_index] = score
            similarity[second_index, first_index] = score

    return similarity


def maybe_split_quad_cluster(
    image_paths: list[Path],
    logger: logging.Logger,
    feature_cache: dict[Path, CachedImageFeatures] | None = None,
) -> list[list[int]] | None:
    if len(image_paths) != 4:
        return None

    records: list[CachedImageFeatures] = []
    for image_path in image_paths:
        record = feature_cache.get(image_path) if feature_cache is not None else None
        if record is None:
            image = load_rgb_image(image_path)
            if image is None:
                return None
            record = CachedImageFeatures(
                path=image_path,
                image=image,
                layout=image_to_layout_vector(image, size=(24, 24)),
                edge=image_to_edge_vector(image, size=(24, 24)),
                color=image_to_color_histogram(image),
                opening=image_to_opening_profile(image),
                structure=image_to_structure_vector(image, size=(24, 24)),
                orb_descriptors=image_to_orb_descriptors(image),
                learned_local_descriptor=None,
            )
        records.append(record)

    similarity = np.zeros((4, 4), dtype=np.float32)
    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            opening_similarity = float(np.dot(records[first_index].opening, records[second_index].opening))
            local_similarity = orb_similarity_score(
                records[first_index].orb_descriptors,
                records[second_index].orb_descriptors,
            )
            similarity_score = (0.7 * opening_similarity) + (0.3 * local_similarity)
            similarity[first_index, second_index] = similarity_score
            similarity[second_index, first_index] = similarity_score

    pairings = [
        [(0, 1), (2, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
    ]
    scored_pairings: list[tuple[float, float, list[tuple[int, int]]]] = []
    for pairing in pairings:
        scores = [float(similarity[left, right]) for left, right in pairing]
        scored_pairings.append((sum(scores), min(scores), pairing))

    scored_pairings.sort(key=lambda item: item[0], reverse=True)
    best_score, best_min_pair, best_pairing = scored_pairings[0]
    second_best_score = scored_pairings[1][0]

    if best_score < 0.65 or best_min_pair < 0.25 or (best_score - second_best_score) < 0.10:
        return None

    logger.info(
        "Refined 4-image room cluster into 2 pairs using viewpoint matching (best=%.3f, second=%.3f)",
        best_score,
        second_best_score,
    )
    return [[left, right] for left, right in best_pairing]


def maybe_refine_broad_viewpoint_cluster(
    image_paths: list[Path],
    similarity_threshold: float,
    linkage: str,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    orb_weight: float,
    structure_weight: float,
    local_descriptor_weight: float,
    logger: logging.Logger,
) -> tuple[list[list[int]], list[int]] | None:
    if len(image_paths) < 5:
        return None

    similarity = viewpoint_similarity_matrix(
        image_paths,
        logger,
        feature_cache=feature_cache,
        orb_weight=orb_weight,
        structure_weight=structure_weight,
        local_descriptor_weight=local_descriptor_weight,
    )
    if similarity is None:
        return None

    if linkage == "graph":
        clusters, noise_indices = graph_connected_components(similarity, similarity_threshold, min_cluster_size=2)
    else:
        distance = 1.0 - similarity
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=max(0.0, 1.0 - similarity_threshold),
        )
        labels = model.fit_predict(distance)

        clusters = []
        noise_indices = []
        for cluster_id in sorted(set(int(label) for label in labels)):
            members = [index for index, label in enumerate(labels) if int(label) == cluster_id]
            if len(members) >= 2:
                clusters.append(members)
            else:
                noise_indices.extend(members)

    if len(clusters) <= 1:
        return None

    logger.info(
        "Refined broad viewpoint cluster of %s images into %s subclusters with %s noise images using %s threshold %.2f",
        len(image_paths),
        len(clusters),
        len(noise_indices),
        linkage,
        similarity_threshold,
    )
    return clusters, noise_indices


def strict_same_corner_item_clusters(
    image_paths: list[Path],
    clip_embeddings: np.ndarray,
    item_features: np.ndarray,
    min_cluster_size: int,
    view_similarity_threshold: float,
    item_similarity_threshold: float,
    semantic_similarity_floor: float,
    strict_cluster_threshold: float,
    linkage: str,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    orb_weight: float,
    structure_weight: float,
    local_descriptor_weight: float,
    logger: logging.Logger,
) -> tuple[list[list[int]], list[int]] | None:
    if len(image_paths) == 0:
        return None

    viewpoint_similarity = viewpoint_similarity_matrix(
        image_paths,
        logger,
        feature_cache=feature_cache,
        orb_weight=orb_weight,
        structure_weight=structure_weight,
        local_descriptor_weight=local_descriptor_weight,
    )
    if viewpoint_similarity is None:
        return None

    semantic_similarity = np.clip(clip_embeddings @ clip_embeddings.T, -1.0, 1.0).astype(np.float32, copy=False)
    item_similarity = np.clip(item_features @ item_features.T, -1.0, 1.0).astype(np.float32, copy=False)

    size = len(image_paths)
    strict_similarity = np.eye(size, dtype=np.float32)
    for first_index in range(size):
        for second_index in range(first_index + 1, size):
            view_score = float(viewpoint_similarity[first_index, second_index])
            item_score = float(item_similarity[first_index, second_index])
            semantic_score = float(semantic_similarity[first_index, second_index])
            if (
                view_score < view_similarity_threshold
                or item_score < item_similarity_threshold
                or semantic_score < semantic_similarity_floor
            ):
                score = 0.0
            else:
                score = (
                    0.50 * view_score
                    + 0.30 * item_score
                    + 0.20 * semantic_score
                )
            strict_similarity[first_index, second_index] = score
            strict_similarity[second_index, first_index] = score

    if len(image_paths) < 2:
        return None

    if linkage == "graph":
        clusters, noise_indices = graph_connected_components(
            strict_similarity,
            strict_cluster_threshold,
            min_cluster_size=min_cluster_size,
        )
    else:
        distance = 1.0 - strict_similarity
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=max(0.0, 1.0 - strict_cluster_threshold),
        )
        labels = model.fit_predict(distance)

        clusters = []
        noise_indices = []
        for cluster_id in sorted(set(int(label) for label in labels)):
            members = [index for index, label in enumerate(labels) if int(label) == cluster_id]
            if len(members) >= min_cluster_size:
                clusters.append(members)
            else:
                noise_indices.extend(members)

    logger.info(
        "Strict same-corner+items refinement produced %s clusters and %s noise images from %s semantic images using %s",
        len(clusters),
        len(noise_indices),
        len(image_paths),
        linkage,
    )
    return clusters, noise_indices


def merge_semantic_subclusters(
    semantic_groups: list[dict],
    clip_embeddings: np.ndarray,
    semantic_merge_threshold: float,
    semantic_viewpoint_similarity: np.ndarray | None,
    merge_view_threshold: float,
    logger: logging.Logger,
) -> tuple[list[np.ndarray], list[dict]]:
    if len(semantic_groups) <= 1:
        return [group["indices"] for group in semantic_groups], []

    parent = list(range(len(semantic_groups)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    centroids: list[np.ndarray] = []
    for group in semantic_groups:
        group_embeddings = clip_embeddings[group["semantic_positions"]]
        centroid = group_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        centroids.append(centroid)

    merge_events: list[dict] = []
    for first_index in range(len(semantic_groups)):
        if semantic_groups[first_index]["frozen"]:
            continue
        for second_index in range(first_index + 1, len(semantic_groups)):
            if semantic_groups[second_index]["frozen"]:
                continue
            semantic_similarity = float(np.dot(centroids[first_index], centroids[second_index]))
            if semantic_viewpoint_similarity is None:
                viewpoint_similarity = 1.0
            else:
                first_positions = semantic_groups[first_index]["semantic_positions"]
                second_positions = semantic_groups[second_index]["semantic_positions"]
                cross_scores = semantic_viewpoint_similarity[np.ix_(first_positions, second_positions)]
                viewpoint_similarity = float(cross_scores.mean()) if cross_scores.size else 0.0

            if semantic_similarity >= semantic_merge_threshold and viewpoint_similarity >= merge_view_threshold:
                logger.info(
                    "Merging same-room subclusters with semantic %.3f and viewpoint %.3f (thresholds %.2f / %.2f)",
                    semantic_similarity,
                    viewpoint_similarity,
                    semantic_merge_threshold,
                    merge_view_threshold,
                )
                merge_events.append(
                    {
                        "left_group_index": first_index,
                        "right_group_index": second_index,
                        "semantic_percent": similarity_to_percent(semantic_similarity),
                        "viewpoint_percent": similarity_to_percent(viewpoint_similarity),
                    }
                )
                union(first_index, second_index)

    merged_groups: dict[int, list[np.ndarray]] = {}
    for group_index, group in enumerate(semantic_groups):
        merged_groups.setdefault(find(group_index), []).append(group["indices"])

    return [np.concatenate(group_parts).astype(int, copy=False) for group_parts in merged_groups.values()], merge_events


def extract_visual_features(
    image_paths: list[Path],
    logger: logging.Logger,
    feature_cache: dict[Path, CachedImageFeatures] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    features: list[np.ndarray] = []
    valid_paths: list[Path] = []

    for image_path in image_paths:
        record = feature_cache.get(image_path) if feature_cache is not None else None
        if record is None:
            image = load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during visual feature extraction: %s", image_path)
                continue
            layout = image_to_layout_vector(image, size=(24, 24))
            edges = image_to_edge_vector(image, size=(24, 24))
            color = image_to_color_histogram(image)
        else:
            layout = record.layout
            edges = record.edge
            color = record.color

        features.append(np.concatenate([layout, edges, color]).astype(np.float32, copy=False))
        valid_paths.append(image_path)

    if not features:
        return np.empty((0, 0), dtype=np.float32), []

    return np.stack(features), valid_paths


def combine_features(
    clip_embeddings: np.ndarray,
    visual_features: np.ndarray,
    semantic_weight: float,
    layout_weight: float,
    edge_weight: float,
    color_weight: float,
) -> np.ndarray:
    if len(clip_embeddings) != len(visual_features):
        raise ValueError("Feature sets must have the same number of rows.")

    visual_layout_dim = 24 * 24
    visual_edge_dim = 24 * 24
    layout_features = visual_features[:, :visual_layout_dim]
    edge_features = visual_features[:, visual_layout_dim : visual_layout_dim + visual_edge_dim]
    color_features = visual_features[:, visual_layout_dim + visual_edge_dim :]

    raw_weights = np.array([semantic_weight, layout_weight, edge_weight, color_weight], dtype=np.float32)
    if np.all(raw_weights <= 0):
        raise ValueError("At least one feature weight must be greater than zero.")

    normalized_weights = raw_weights / raw_weights.sum()
    weighted_parts = [
        clip_embeddings * normalized_weights[0],
        layout_features * normalized_weights[1],
        edge_features * normalized_weights[2],
        color_features * normalized_weights[3],
    ]
    return l2_normalize(np.concatenate(weighted_parts, axis=1).astype(np.float32, copy=False))


def embed_images(
    image_paths: list[Path],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[Path]]:
    clip, clip_source = load_clip_module()

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded CLIP from %s", clip_source)
    model, preprocess = clip.load(model_name, device=device, jit=False, download_root=str(cache_dir))
    model.eval()

    embedded_paths: list[Path] = []
    batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []
            batch_valid_paths = []

            for image_path in batch_paths:
                cached = feature_cache.get(image_path) if feature_cache is not None else None
                image = cached.image if cached is not None else load_rgb_image(image_path)
                if image is None:
                    logger.warning("Skipping unreadable image: %s", image_path)
                    continue
                batch_tensors.append(preprocess(image))
                batch_valid_paths.append(image_path)

            if not batch_tensors:
                continue

            batch_tensor = torch.stack(batch_tensors).to(device)
            batch_features = model.encode_image(batch_tensor).float().cpu().numpy()
            batches.append(batch_features)
            embedded_paths.extend(batch_valid_paths)
            logger.info("Embedded %s/%s images", len(embedded_paths), len(image_paths))

    if not batches:
        return np.empty((0, 0), dtype=np.float32), []

    return l2_normalize(np.concatenate(batches, axis=0).astype(np.float32, copy=False)), embedded_paths


def extract_clip_item_features(
    image_paths: list[Path],
    model_name: str,
    batch_size: int,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), []

    clip, model, preprocess = load_clip_runtime(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        logger=logger,
    )

    with torch.no_grad():
        prompt_tokens = clip.tokenize(prompt_texts).to(device)
        prompt_features = model.encode_text(prompt_tokens).float()
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

    valid_paths: list[Path] = []
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = []
            batch_valid_paths = []
            for image_path in batch_paths:
                cached = feature_cache.get(image_path) if feature_cache is not None else None
                image = cached.image if cached is not None else load_rgb_image(image_path)
                if image is None:
                    logger.warning("Skipping unreadable image during item signature extraction: %s", image_path)
                    continue
                batch_tensors.append(preprocess(image))
                batch_valid_paths.append(image_path)

            if not batch_tensors:
                continue

            batch_tensor = torch.stack(batch_tensors).to(device)
            image_features = model.encode_image(batch_tensor).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ prompt_features.T
            prompt_distribution = torch.softmax(similarity * 10.0, dim=1)
            batches.append(prompt_distribution.cpu().numpy().astype(np.float32, copy=False))
            valid_paths.extend(batch_valid_paths)

    if not batches:
        return np.empty((0, 0), dtype=np.float32), []

    return l2_normalize(np.concatenate(batches, axis=0).astype(np.float32, copy=False)), valid_paths


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_epsilon: float,
    max_cluster_size: int | None = None,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) < min_cluster_size:
        return np.full(len(embeddings), -1, dtype=int)

    clusterer = HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=max(0.0, cluster_epsilon),
        cluster_selection_method="eom",
        max_cluster_size=max_cluster_size,
        n_jobs=1,
        copy=True,
    )
    return clusterer.fit_predict(embeddings).astype(int, copy=False)


def cluster_same_corner_groups(
    image_paths: list[Path],
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    min_cluster_size: int,
    min_samples: int,
    cluster_epsilon: float,
    view_max_cluster_size: int | None,
    view_similarity_threshold: float,
    semantic_merge_threshold: float,
    merge_view_threshold: float,
    strict_same_corner_items: bool,
    item_similarity_threshold: float,
    strict_cluster_threshold: float,
    semantic_similarity_floor: float,
    view_linkage: str,
    strict_linkage: str,
    orb_weight: float,
    structure_weight: float,
    local_descriptor_weight: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[dict]]:
    if len(clip_embeddings) == 0:
        return np.array([], dtype=int), []

    semantic_labels = cluster_embeddings(
        embeddings=clip_embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_epsilon=0.0,
    )
    final_labels = np.full(len(clip_embeddings), -1, dtype=int)
    next_label = 0

    semantic_cluster_ids = sorted({int(label) for label in semantic_labels if int(label) != -1})
    if not semantic_cluster_ids:
        logger.info("No stable first-stage semantic clusters found; falling back to one-stage hybrid clustering.")
        return (
            cluster_embeddings(
                embeddings=hybrid_embeddings,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_epsilon=cluster_epsilon,
                max_cluster_size=view_max_cluster_size,
            ),
            [],
        )

    merge_events: list[dict] = []

    for semantic_cluster_id in semantic_cluster_ids:
        semantic_indices = np.where(semantic_labels == semantic_cluster_id)[0]
        semantic_count = len(semantic_indices)
        logger.info("First-stage semantic cluster %s contains %s images", semantic_cluster_id, semantic_count)

        if semantic_count < min_cluster_size:
            continue

        if strict_same_corner_items:
            if item_features is None or len(item_features) != len(clip_embeddings):
                raise ValueError("Strict same-corner+items mode requires item features for every embedded image.")

            refined_pairs = maybe_split_quad_cluster(
                [image_paths[index] for index in semantic_indices],
                logger,
                feature_cache=feature_cache,
            )
            if refined_pairs is not None:
                for pair_indices in refined_pairs:
                    cluster_indices = semantic_indices[pair_indices]
                    final_labels[cluster_indices] = next_label
                    logger.info(
                        "Semantic cluster %s -> strict paired cluster %s contains %s images",
                        semantic_cluster_id,
                        next_label,
                        len(cluster_indices),
                    )
                    next_label += 1
                continue

            strict_refinement = strict_same_corner_item_clusters(
                image_paths=[image_paths[index] for index in semantic_indices],
                clip_embeddings=clip_embeddings[semantic_indices],
                item_features=item_features[semantic_indices],
                min_cluster_size=min_cluster_size,
                view_similarity_threshold=view_similarity_threshold,
                item_similarity_threshold=item_similarity_threshold,
                semantic_similarity_floor=semantic_similarity_floor,
                strict_cluster_threshold=strict_cluster_threshold,
                linkage=strict_linkage,
                feature_cache=feature_cache,
                orb_weight=orb_weight,
                structure_weight=structure_weight,
                local_descriptor_weight=local_descriptor_weight,
                logger=logger,
            )
            if strict_refinement is None:
                continue

            strict_clusters, strict_noise = strict_refinement
            for strict_cluster in strict_clusters:
                cluster_indices = semantic_indices[strict_cluster]
                final_labels[cluster_indices] = next_label
                logger.info(
                    "Semantic cluster %s -> strict cluster %s contains %s images",
                    semantic_cluster_id,
                    next_label,
                    len(cluster_indices),
                )
                next_label += 1
            if strict_noise:
                final_labels[semantic_indices[strict_noise]] = -1
            continue

        semantic_groups: list[dict] = []
        semantic_viewpoint_similarity = viewpoint_similarity_matrix(
            [image_paths[index] for index in semantic_indices],
            logger,
            feature_cache=feature_cache,
            orb_weight=orb_weight,
            structure_weight=structure_weight,
            local_descriptor_weight=local_descriptor_weight,
        )
        local_labels = cluster_embeddings(
            embeddings=hybrid_embeddings[semantic_indices],
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_epsilon=cluster_epsilon,
            max_cluster_size=view_max_cluster_size,
        )
        local_cluster_ids = sorted({int(label) for label in local_labels if int(label) != -1})

        non_noise_count = int(np.sum(local_labels != -1))
        if not local_cluster_ids or (len(local_cluster_ids) == 1 and non_noise_count == semantic_count):
            refined_pairs = maybe_split_quad_cluster(
                [image_paths[index] for index in semantic_indices],
                logger,
                feature_cache=feature_cache,
            )
            if refined_pairs is not None:
                for pair_indices in refined_pairs:
                    semantic_groups.append(
                        {
                            "indices": semantic_indices[pair_indices].astype(int, copy=False),
                            "semantic_positions": np.asarray(pair_indices, dtype=int),
                            "frozen": True,
                        }
                    )

        if semantic_groups:
            pass
        elif not local_cluster_ids:
            logger.info(
                "Semantic cluster %s could not be split by viewpoint; keeping it as one cluster.",
                semantic_cluster_id,
            )
            semantic_groups.append(
                {
                    "indices": semantic_indices.astype(int, copy=False),
                    "semantic_positions": np.arange(semantic_count, dtype=int),
                    "frozen": False,
                }
            )
        else:
            for local_cluster_id in local_cluster_ids:
                local_mask = local_labels == local_cluster_id
                local_indices = semantic_indices[local_mask]
                local_positions = np.where(local_mask)[0]

                refined_large_cluster = maybe_refine_broad_viewpoint_cluster(
                    [image_paths[index] for index in local_indices],
                    similarity_threshold=view_similarity_threshold,
                    linkage=view_linkage,
                    feature_cache=feature_cache,
                    orb_weight=orb_weight,
                    structure_weight=structure_weight,
                    local_descriptor_weight=local_descriptor_weight,
                    logger=logger,
                )
                if refined_large_cluster is not None:
                    refined_subclusters, refined_noise = refined_large_cluster
                    for refined_group in refined_subclusters:
                        semantic_groups.append(
                            {
                                "indices": local_indices[refined_group].astype(int, copy=False),
                                "semantic_positions": local_positions[refined_group].astype(int, copy=False),
                                "frozen": False,
                            }
                        )
                    if refined_noise:
                        final_labels[local_indices[refined_noise]] = -1
                    continue

                semantic_groups.append(
                    {
                        "indices": local_indices.astype(int, copy=False),
                        "semantic_positions": local_positions.astype(int, copy=False),
                        "frozen": False,
                    }
                )

        merged_groups, cluster_merge_events = merge_semantic_subclusters(
            semantic_groups=semantic_groups,
            clip_embeddings=clip_embeddings[semantic_indices],
            semantic_merge_threshold=semantic_merge_threshold,
            semantic_viewpoint_similarity=semantic_viewpoint_similarity,
            merge_view_threshold=merge_view_threshold,
            logger=logger,
        )
        for event in cluster_merge_events:
            event["semantic_cluster_id"] = int(semantic_cluster_id)
        merge_events.extend(cluster_merge_events)

        for merged_group in merged_groups:
            final_labels[merged_group] = next_label
            logger.info(
                "Semantic cluster %s -> viewpoint cluster %s contains %s images",
                semantic_cluster_id,
                next_label,
                len(merged_group),
            )
            next_label += 1

    return final_labels, merge_events


def similarity_to_percent(score: float) -> float:
    return round(float(np.clip(score, 0.0, 1.0)) * 100.0, 2)


def compute_similarity_matrices(
    image_paths: list[Path],
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    orb_weight: float,
    structure_weight: float,
    local_descriptor_weight: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    semantic_similarity = np.clip(clip_embeddings @ clip_embeddings.T, -1.0, 1.0).astype(np.float32, copy=False)
    hybrid_similarity = np.clip(hybrid_embeddings @ hybrid_embeddings.T, -1.0, 1.0).astype(np.float32, copy=False)
    viewpoint_similarity = viewpoint_similarity_matrix(
        image_paths,
        logger,
        feature_cache=feature_cache,
        orb_weight=orb_weight,
        structure_weight=structure_weight,
        local_descriptor_weight=local_descriptor_weight,
    )
    if viewpoint_similarity is None:
        viewpoint_similarity = np.eye(len(image_paths), dtype=np.float32)
    else:
        viewpoint_similarity = np.clip(viewpoint_similarity, -1.0, 1.0).astype(np.float32, copy=False)

    item_similarity: np.ndarray | None = None
    if item_features is not None and len(item_features) == len(image_paths):
        item_similarity = np.clip(item_features @ item_features.T, -1.0, 1.0).astype(np.float32, copy=False)

    return semantic_similarity, hybrid_similarity, viewpoint_similarity, item_similarity


def build_image_quality_report(image: Image.Image | None) -> dict:
    if image is None:
        return {
            "blur_score": None,
            "brightness_mean": None,
            "contrast_std": None,
            "issues": ["unreadable"],
            "severe_quality_issue": True,
        }

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())

    issues: list[str] = []
    if blur_score < 45.0:
        issues.append("blurry")
    if contrast_std < 18.0:
        issues.append("low_contrast")
    if brightness_mean < 40.0:
        issues.append("underexposed")
    elif brightness_mean > 235.0:
        issues.append("overexposed")

    return {
        "blur_score": round(blur_score, 2),
        "brightness_mean": round(brightness_mean, 2),
        "contrast_std": round(contrast_std, 2),
        "issues": issues,
        "severe_quality_issue": any(issue in {"blurry", "underexposed", "overexposed"} for issue in issues),
    }


def collect_image_quality_reports(
    image_paths: list[Path],
    feature_cache: dict[Path, CachedImageFeatures] | None,
) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for image_path in image_paths:
        cached = feature_cache.get(image_path) if feature_cache is not None else None
        reports[image_path.name] = build_image_quality_report(cached.image if cached is not None else load_rgb_image(image_path))
    return reports


def mean_top_scores(scores: np.ndarray, top_k: int) -> float:
    if scores.size == 0:
        return 0.0
    sorted_scores = np.sort(scores.astype(np.float32, copy=False))
    return float(sorted_scores[-max(1, top_k) :].mean())


def deduplicate_reasons(reasons: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduplicated.append(reason)
    return deduplicated


def finalize_noise_labels(
    image_paths: list[Path],
    labels: np.ndarray,
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    strict_same_corner_items: bool,
    view_similarity_threshold: float,
    item_similarity_threshold: float,
    semantic_similarity_floor: float,
    min_cluster_size: int,
    orb_weight: float,
    structure_weight: float,
    local_descriptor_weight: float,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, dict], dict[str, dict], list[dict]]:
    if not image_paths:
        return np.array([], dtype=int), {}, {}, []

    final_labels = labels.astype(int, copy=True)
    quality_reports = collect_image_quality_reports(image_paths, feature_cache)
    semantic_similarity, hybrid_similarity, viewpoint_similarity, item_similarity = compute_similarity_matrices(
        image_paths=image_paths,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=hybrid_embeddings,
        item_features=item_features,
        feature_cache=feature_cache,
        orb_weight=orb_weight,
        structure_weight=structure_weight,
        local_descriptor_weight=local_descriptor_weight,
        logger=logger,
    )

    semantic_guardrail = max(0.80, semantic_similarity_floor - 0.08)
    viewpoint_guardrail = max(0.20, view_similarity_threshold - 0.08)
    item_guardrail = max(0.70, item_similarity_threshold - 0.08)
    support_requirement = max(1, min_cluster_size - 1)

    noise_details: dict[str, dict] = {}
    reassigned_images: list[dict] = []

    for image_index, image_path in enumerate(image_paths):
        if int(final_labels[image_index]) != -1:
            continue

        quality_report = quality_reports[image_path.name]
        reasons = list(quality_report["issues"])
        best_candidate: dict | None = None

        candidate_labels = sorted({int(label) for label in final_labels if int(label) != -1})
        for cluster_label in candidate_labels:
            cluster_indices = np.where(final_labels == cluster_label)[0]
            if cluster_indices.size == 0:
                continue

            semantic_scores = np.clip(semantic_similarity[image_index, cluster_indices], 0.0, 1.0)
            hybrid_scores = np.clip(hybrid_similarity[image_index, cluster_indices], 0.0, 1.0)
            viewpoint_scores = np.clip(viewpoint_similarity[image_index, cluster_indices], 0.0, 1.0)

            item_scores: np.ndarray | None = None
            if item_similarity is not None:
                item_scores = np.clip(item_similarity[image_index, cluster_indices], 0.0, 1.0)

            top_k = min(2, cluster_indices.size)
            semantic_top = mean_top_scores(semantic_scores, top_k)
            hybrid_top = mean_top_scores(hybrid_scores, top_k)
            viewpoint_top = mean_top_scores(viewpoint_scores, top_k)
            item_top = mean_top_scores(item_scores, top_k) if item_scores is not None else None

            support_mask = (semantic_scores >= semantic_guardrail) & (viewpoint_scores >= viewpoint_guardrail)
            if strict_same_corner_items and item_scores is not None:
                support_mask &= item_scores >= item_guardrail
                compatibility_score = (
                    0.45 * viewpoint_top
                    + 0.30 * semantic_top
                    + 0.25 * float(item_top or 0.0)
                )
            else:
                compatibility_score = (
                    0.50 * hybrid_top
                    + 0.30 * viewpoint_top
                    + 0.20 * semantic_top
                )

            passes_guardrails = (
                semantic_top >= semantic_guardrail
                and viewpoint_top >= viewpoint_guardrail
                and int(np.sum(support_mask)) >= support_requirement
            )
            if strict_same_corner_items and item_top is not None:
                passes_guardrails = passes_guardrails and item_top >= item_guardrail
            if quality_report["severe_quality_issue"]:
                passes_guardrails = passes_guardrails and int(np.sum(support_mask)) >= max(2, support_requirement)

            candidate = {
                "cluster_id": cluster_label,
                "semantic_top": semantic_top,
                "hybrid_top": hybrid_top,
                "viewpoint_top": viewpoint_top,
                "item_top": item_top,
                "support_count": int(np.sum(support_mask)),
                "compatibility_score": compatibility_score,
                "passes_guardrails": passes_guardrails,
            }
            if best_candidate is None or candidate["compatibility_score"] > best_candidate["compatibility_score"]:
                best_candidate = candidate

        if best_candidate is not None and best_candidate["passes_guardrails"]:
            final_labels[image_index] = int(best_candidate["cluster_id"])
            reassigned_images.append(
                {
                    "image": image_path.name,
                    "cluster_id": int(best_candidate["cluster_id"]),
                    "compatibility_percent": similarity_to_percent(best_candidate["compatibility_score"]),
                    "quality_issues": quality_report["issues"],
                }
            )
            logger.info(
                "Reassigned noise image %s to cluster %s (compatibility %.2f)",
                image_path.name,
                best_candidate["cluster_id"],
                best_candidate["compatibility_score"],
            )
            continue

        if best_candidate is None:
            reasons.append("no_cluster_candidates")
            best_payload = None
        else:
            if best_candidate["semantic_top"] < semantic_guardrail:
                reasons.append("semantic_mismatch")
            if best_candidate["viewpoint_top"] < viewpoint_guardrail:
                reasons.append("viewpoint_mismatch")
            if strict_same_corner_items and best_candidate["item_top"] is not None:
                if best_candidate["item_top"] < item_guardrail:
                    reasons.append("item_mismatch")
            if int(best_candidate["support_count"]) < support_requirement:
                reasons.append("insufficient_cluster_support")
            if float(best_candidate["compatibility_score"]) < 0.60:
                reasons.append("low_cluster_compatibility")
            best_payload = {
                "cluster_id": int(best_candidate["cluster_id"]),
                "semantic_percent": similarity_to_percent(best_candidate["semantic_top"]),
                "hybrid_percent": similarity_to_percent(best_candidate["hybrid_top"]),
                "viewpoint_percent": similarity_to_percent(best_candidate["viewpoint_top"]),
                "compatibility_percent": similarity_to_percent(best_candidate["compatibility_score"]),
                "support_count": int(best_candidate["support_count"]),
            }
            if best_candidate["item_top"] is not None:
                best_payload["item_percent"] = similarity_to_percent(float(best_candidate["item_top"]))

        noise_details[image_path.name] = {
            "reasons": deduplicate_reasons(reasons),
            "quality": quality_report,
            "best_candidate": best_payload,
        }

    return final_labels, noise_details, quality_reports, reassigned_images


def build_match_scores_payload(
    image_paths: list[Path],
    labels: np.ndarray,
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    strict_same_corner_items: bool,
    view_similarity_threshold: float,
    item_similarity_threshold: float,
    semantic_similarity_floor: float,
    orb_weight: float,
    structure_weight: float,
    local_descriptor_weight: float,
    quality_reports: dict[str, dict] | None,
    noise_details: dict[str, dict] | None,
    reassigned_images: list[dict] | None,
    logger: logging.Logger,
) -> dict:
    if not image_paths:
        return {"mode": "strict" if strict_same_corner_items else "default", "images": []}

    semantic_similarity, hybrid_similarity, viewpoint_similarity, item_similarity = compute_similarity_matrices(
        image_paths=image_paths,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=hybrid_embeddings,
        item_features=item_features,
        feature_cache=feature_cache,
        orb_weight=orb_weight,
        structure_weight=structure_weight,
        local_descriptor_weight=local_descriptor_weight,
        logger=logger,
    )
    reassigned_lookup = {item["image"]: item for item in (reassigned_images or [])}

    images_payload: list[dict] = []
    for image_index, image_path in enumerate(image_paths):
        matches: list[dict] = []
        for other_index, other_path in enumerate(image_paths):
            if image_index == other_index:
                continue

            semantic_score = float(np.clip(semantic_similarity[image_index, other_index], 0.0, 1.0))
            hybrid_score = float(np.clip(hybrid_similarity[image_index, other_index], 0.0, 1.0))
            viewpoint_score = float(np.clip(viewpoint_similarity[image_index, other_index], 0.0, 1.0))
            item_score = None
            if item_similarity is not None:
                item_score = float(np.clip(item_similarity[image_index, other_index], 0.0, 1.0))

            if strict_same_corner_items and item_score is not None:
                match_score = (
                    0.50 * viewpoint_score
                    + 0.30 * item_score
                    + 0.20 * semantic_score
                )
                passes_thresholds = (
                    viewpoint_score >= view_similarity_threshold
                    and item_score >= item_similarity_threshold
                    and semantic_score >= semantic_similarity_floor
                )
            else:
                match_score = (
                    0.55 * hybrid_score
                    + 0.30 * viewpoint_score
                    + 0.15 * semantic_score
                )
                passes_thresholds = None

            match_payload = {
                "image": other_path.name,
                "match_percent": similarity_to_percent(match_score),
                "same_cluster": bool(int(labels[image_index]) == int(labels[other_index]) and int(labels[image_index]) != -1),
                "semantic_percent": similarity_to_percent(semantic_score),
                "hybrid_percent": similarity_to_percent(hybrid_score),
                "viewpoint_percent": similarity_to_percent(viewpoint_score),
            }
            if item_score is not None:
                match_payload["item_percent"] = similarity_to_percent(item_score)
            if passes_thresholds is not None:
                match_payload["passes_strict_thresholds"] = passes_thresholds

            matches.append(match_payload)

        matches.sort(key=lambda item: item["match_percent"], reverse=True)
        images_payload.append(
            {
                "image": image_path.name,
                "cluster_id": int(labels[image_index]),
                "matches": matches,
                "quality": quality_reports.get(image_path.name) if quality_reports is not None else None,
                "noise_analysis": noise_details.get(image_path.name) if noise_details is not None else None,
                "reassigned_from_noise": reassigned_lookup.get(image_path.name),
            }
        )

    return {
        "mode": "strict" if strict_same_corner_items else "default",
        "images": images_payload,
        "reassigned_from_noise": list(reassigned_lookup.values()),
    }


def resolve_output_dir(output_dir: Path, overwrite: bool, logger: logging.Logger) -> Path:
    if not output_dir.exists():
        return output_dir

    if overwrite:
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir.parent / f"{output_dir.name}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_dir.parent / f"{output_dir.name}_{timestamp}_{suffix}"
        suffix += 1

    logger.info("Output folder %s already exists; writing to %s instead.", output_dir, candidate)
    return candidate


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and overwrite is disabled: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def save_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    thumb_size: tuple[int, int] = (240, 180),
    columns: int = 4,
    feature_cache: dict[Path, CachedImageFeatures] | None = None,
) -> Path | None:
    thumbnails: list[Image.Image] = []
    for image_path in image_paths:
        cached = feature_cache.get(image_path) if feature_cache is not None else None
        image = cached.image if cached is not None else load_rgb_image(image_path)
        if image is None:
            continue
        thumbnails.append(ImageOps.fit(image, thumb_size, Image.Resampling.BICUBIC))

    if not thumbnails:
        return None

    rows = (len(thumbnails) + columns - 1) // columns
    padding = 12
    sheet_width = columns * thumb_size[0] + (columns + 1) * padding
    sheet_height = rows * thumb_size[1] + (rows + 1) * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(248, 248, 248))

    for index, thumbnail in enumerate(thumbnails):
        row = index // columns
        column = index % columns
        left = padding + column * (thumb_size[0] + padding)
        top = padding + row * (thumb_size[1] + padding)
        sheet.paste(thumbnail, (left, top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return output_path


def build_resolved_run_settings(
    args: argparse.Namespace,
    input_dir: Path,
    requested_output_dir: Path,
    output_dir: Path,
    resolved_device: str,
) -> dict:
    return {
        "input_dir": str(input_dir),
        "requested_output_dir": str(requested_output_dir),
        "output_dir": str(output_dir),
        "preset": args.preset,
        "model": args.model,
        "batch_size": max(1, args.batch_size),
        "min_cluster_size": int(args.min_cluster_size),
        "min_samples": int(args.min_samples),
        "cluster_epsilon": float(args.cluster_epsilon),
        "semantic_weight": float(args.semantic_weight),
        "layout_weight": float(args.layout_weight),
        "edge_weight": float(args.edge_weight),
        "color_weight": float(args.color_weight),
        "view_max_cluster_size": args.view_max_cluster_size,
        "view_similarity_threshold": float(args.view_similarity_threshold),
        "semantic_merge_threshold": float(args.semantic_merge_threshold),
        "strict_same_corner_items": bool(args.strict_same_corner_items),
        "prompt_set": args.prompt_set,
        "item_similarity_threshold": float(args.item_similarity_threshold),
        "strict_cluster_threshold": float(args.strict_cluster_threshold),
        "semantic_similarity_floor": float(args.semantic_similarity_floor),
        "merge_view_threshold": float(args.merge_view_threshold),
        "view_linkage": args.view_linkage,
        "strict_linkage": args.strict_linkage,
        "orb_weight": float(args.orb_weight),
        "structure_weight": float(args.structure_weight),
        "local_descriptor_mode": args.local_descriptor_mode,
        "local_descriptor_weight": float(args.local_descriptor_weight),
        "device": args.device,
        "resolved_device": resolved_device,
        "overwrite": bool(args.overwrite),
        "skip_contact_sheets": bool(args.skip_contact_sheets),
        "skip_html_summary": bool(args.skip_html_summary),
    }


def write_run_manifest(
    output_dir: Path,
    settings: dict,
    result: dict,
    discovered_images: int,
    processed_images: int,
) -> dict:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "summary": {
            "discovered_images": discovered_images,
            "processed_images": processed_images,
            "cluster_count": len(result.get("clusters", [])),
            "noise_count": len(result.get("noise", [])),
            "reassigned_from_noise_count": len(result.get("reassigned_from_noise", [])),
        },
        "result": result,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def get_prompt_texts(prompt_set: str) -> list[str]:
    prompts = PROMPT_SET_CONFIGS.get(prompt_set)
    if not isinstance(prompts, list) or not prompts:
        prompts = DEFAULT_PROMPT_SET_CONFIGS["real_estate"]
    return [str(prompt) for prompt in prompts]


def validate_runtime_setup(
    prompt_set: str,
    preset: str,
    local_descriptor_mode: str,
    logger: logging.Logger,
) -> dict:
    clip_module, clip_source = load_clip_module()
    prompt_texts = get_prompt_texts(prompt_set)
    report = {
        "clip_source": clip_source,
        "clip_has_load": bool(hasattr(clip_module, "load")),
        "opencv_available": bool(hasattr(cv2, "ORB_create")),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "preset": preset,
        "prompt_set": prompt_set,
        "prompt_count": len(prompt_texts),
        "local_descriptor_mode": local_descriptor_mode,
    }
    logger.info(
        "Setup validation: preset=%s prompt_set=%s prompts=%s local_descriptor_mode=%s clip=%s cuda=%s",
        preset,
        prompt_set,
        len(prompt_texts),
        local_descriptor_mode,
        clip_source,
        torch.cuda.is_available(),
    )
    return report


def write_html_summary(
    output_dir: Path,
    result: dict,
    manifest: dict,
    match_payload: dict,
) -> Path:
    image_matches = {entry["image"]: entry for entry in match_payload.get("images", [])}
    cluster_sections: list[str] = []
    for cluster in result.get("clusters", []):
        images = cluster.get("images", [])
        representative = images[0] if images else None
        near_misses: list[dict] = []
        if representative is not None:
            representative_entry = image_matches.get(representative, {})
            for match in representative_entry.get("matches", []):
                if match.get("same_cluster"):
                    continue
                near_misses.append(match)
                if len(near_misses) == 3:
                    break

        contact_sheet = cluster.get("contact_sheet")
        contact_html = (
            f'<img src="{escape(contact_sheet)}" alt="{escape(contact_sheet)}" style="max-width: 100%; border-radius: 8px;" />'
            if contact_sheet
            else "<div>No contact sheet generated.</div>"
        )
        near_miss_html = "".join(
            f"<li>{escape(item['image'])} ({item['match_percent']}%)</li>" for item in near_misses
        ) or "<li>None</li>"
        images_html = "".join(f"<li>{escape(image_name)}</li>" for image_name in images)
        cluster_sections.append(
            """
            <section class="cluster-card">
              <h2>Cluster {cluster_id}</h2>
              {contact_html}
              <p><strong>Representative:</strong> {representative}</p>
              <p><strong>Count:</strong> {count}</p>
              <div class="two-col">
                <div>
                  <h3>Images</h3>
                  <ul>{images_html}</ul>
                </div>
                <div>
                  <h3>Top Near Misses</h3>
                  <ul>{near_miss_html}</ul>
                </div>
              </div>
            </section>
            """.format(
                cluster_id=cluster["cluster_id"],
                contact_html=contact_html,
                representative=escape(representative or "None"),
                count=len(images),
                images_html=images_html,
                near_miss_html=near_miss_html,
            )
        )

    noise_cards = []
    for noise_item in result.get("noise_details", []):
        reasons = ", ".join(noise_item.get("reasons", [])) or "unspecified"
        best_candidate = noise_item.get("best_candidate")
        candidate_text = "None"
        if best_candidate is not None:
            candidate_text = (
                f"cluster {best_candidate['cluster_id']} "
                f"(compatibility {best_candidate['compatibility_percent']}%)"
            )
        noise_cards.append(
            f"<li><strong>{escape(noise_item['image'])}</strong>: {escape(reasons)}. Best candidate: {escape(candidate_text)}</li>"
        )
    noise_html = "".join(noise_cards) or "<li>No noise images.</li>"
    noise_contact = result.get("noise_contact_sheet")
    noise_contact_html = (
        f'<img src="{escape(noise_contact)}" alt="{escape(noise_contact)}" style="max-width: 100%; border-radius: 8px;" />'
        if noise_contact
        else "<div>No noise contact sheet generated.</div>"
    )

    settings_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in manifest.get("settings", {}).items()
    )
    merge_events_html = "".join(
        f"<li>Semantic cluster {event['semantic_cluster_id']}: groups {event['left_group_index']} + {event['right_group_index']} "
        f"(semantic {event['semantic_percent']}%, viewpoint {event['viewpoint_percent']}%)</li>"
        for event in result.get("merge_events", [])
    ) or "<li>No merge-back events recorded.</li>"
    skipped_html = "".join(f"<li>{escape(name)}</li>" for name in result.get("skipped_images", [])) or "<li>None</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Clustering Summary</title>
  <style>
    body {{ font-family: Segoe UI, sans-serif; margin: 0; background: #f4f6f8; color: #1f2933; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 60px; }}
    h1, h2, h3 {{ margin-bottom: 12px; }}
    .hero {{ background: linear-gradient(135deg, #0f4c5c, #3a506b); color: white; border-radius: 18px; padding: 28px; margin-bottom: 24px; }}
    .hero p {{ margin: 6px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
    .cluster-card, .panel {{ background: white; border-radius: 16px; padding: 20px; box-shadow: 0 12px 30px rgba(15, 76, 92, 0.08); }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    ul {{ margin-top: 8px; padding-left: 20px; }}
    @media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Clustering Summary</h1>
      <p><strong>Generated:</strong> {escape(manifest.get('generated_at', 'unknown'))}</p>
      <p><strong>Preset:</strong> {escape(str(manifest.get('settings', {}).get('preset')))}</p>
      <p><strong>Prompt set:</strong> {escape(str(manifest.get('settings', {}).get('prompt_set')))}</p>
      <p><strong>Clusters:</strong> {len(result.get('clusters', []))} | <strong>Noise:</strong> {len(result.get('noise', []))}</p>
    </section>

    <section class="grid">
      <section class="panel">
        <h2>Run Settings</h2>
        <table>{settings_rows}</table>
      </section>
      <section class="panel">
        <h2>Merge Events</h2>
        <ul>{merge_events_html}</ul>
      </section>
      <section class="panel">
        <h2>Skipped Images</h2>
        <ul>{skipped_html}</ul>
      </section>
      <section class="panel">
        <h2>Noise Review</h2>
        {noise_contact_html}
        <ul>{noise_html}</ul>
      </section>
    </section>

    <div class="grid" style="margin-top: 24px;">
      {''.join(cluster_sections)}
    </div>
  </main>
</body>
</html>
"""
    summary_path = output_dir / "summary.html"
    summary_path.write_text(html, encoding="utf-8")
    return summary_path


def copy_clustered_images(
    image_paths: list[Path],
    labels: np.ndarray,
    output_dir: Path,
    noise_details: dict[str, dict] | None = None,
    reassigned_images: list[dict] | None = None,
    generate_contact_sheets: bool = True,
    feature_cache: dict[Path, CachedImageFeatures] | None = None,
    merge_events: list[dict] | None = None,
    skipped_images: list[str] | None = None,
) -> dict:
    clusters_payload: list[dict] = []
    noise_images: list[str] = []
    noise_details_payload: list[dict] = []
    noise_contact_sheet: str | None = None

    unique_labels = sorted(set(int(label) for label in labels))
    for label in unique_labels:
        members = [path for path, cluster_label in zip(image_paths, labels) if int(cluster_label) == label]
        if not members:
            continue

        if label == -1:
            noise_dir = output_dir / "noise"
            noise_dir.mkdir(parents=True, exist_ok=True)
            for image_path in members:
                shutil.copy2(image_path, noise_dir / image_path.name)
                noise_images.append(image_path.name)
                if noise_details is not None and image_path.name in noise_details:
                    noise_details_payload.append(
                        {
                            "image": image_path.name,
                            **noise_details[image_path.name],
                        }
                    )
            if generate_contact_sheets:
                contact_sheet_path = save_contact_sheet(
                    members,
                    output_dir / "noise_contact.jpg",
                    feature_cache=feature_cache,
                )
                if contact_sheet_path is not None:
                    noise_contact_sheet = contact_sheet_path.name
            continue

        cluster_dir = output_dir / f"cluster_{label}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for image_path in members:
            shutil.copy2(image_path, cluster_dir / image_path.name)

        contact_sheet_name: str | None = None
        if generate_contact_sheets:
            contact_sheet_path = save_contact_sheet(
                members,
                output_dir / f"cluster_{label}_contact.jpg",
                feature_cache=feature_cache,
            )
            if contact_sheet_path is not None:
                contact_sheet_name = contact_sheet_path.name

        clusters_payload.append(
            {
                "cluster_id": label,
                "images": [path.name for path in members],
                "contact_sheet": contact_sheet_name,
            }
        )

    result = {
        "clusters": clusters_payload,
        "noise": noise_images,
        "noise_details": noise_details_payload,
        "reassigned_from_noise": reassigned_images or [],
        "noise_contact_sheet": noise_contact_sheet,
        "merge_events": merge_events or [],
        "skipped_images": skipped_images or [],
    }
    (output_dir / "clusters.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_match_scores(
    image_paths: list[Path],
    labels: np.ndarray,
    clip_embeddings: np.ndarray,
    hybrid_embeddings: np.ndarray,
    item_features: np.ndarray | None,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    strict_same_corner_items: bool,
    view_similarity_threshold: float,
    item_similarity_threshold: float,
    semantic_similarity_floor: float,
    orb_weight: float,
    structure_weight: float,
    local_descriptor_weight: float,
    quality_reports: dict[str, dict] | None,
    noise_details: dict[str, dict] | None,
    reassigned_images: list[dict] | None,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    payload = build_match_scores_payload(
        image_paths=image_paths,
        labels=labels,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=hybrid_embeddings,
        item_features=item_features,
        feature_cache=feature_cache,
        strict_same_corner_items=strict_same_corner_items,
        view_similarity_threshold=view_similarity_threshold,
        item_similarity_threshold=item_similarity_threshold,
        semantic_similarity_floor=semantic_similarity_floor,
        orb_weight=orb_weight,
        structure_weight=structure_weight,
        local_descriptor_weight=local_descriptor_weight,
        quality_reports=quality_reports,
        noise_details=noise_details,
        reassigned_images=reassigned_images,
        logger=logger,
    )
    (output_dir / "match_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    args = apply_preset_defaults(args, sys.argv[1:])
    logger = setup_logging()

    if args.validate_setup:
        report = validate_runtime_setup(
            prompt_set=args.prompt_set,
            preset=args.preset,
            local_descriptor_mode=args.local_descriptor_mode,
            logger=logger,
        )
        print(json.dumps(report, indent=2))
        return

    input_dir = Path(args.input).resolve()
    requested_output_dir = Path(args.output).resolve()
    output_dir = resolve_output_dir(requested_output_dir, overwrite=args.overwrite, logger=logger)
    cache_dir = Path(".clip-cache").resolve()
    device = resolve_device(args.device)
    resolved_settings = build_resolved_run_settings(args, input_dir, requested_output_dir, output_dir, device)

    logger.info("Reading images from %s", input_dir)
    image_paths = discover_images(input_dir)
    if not image_paths:
        raise SystemExit("No supported images found in input folder.")

    logger.info("Found %s images", len(image_paths))
    feature_cache, cached_paths, skipped_images = build_feature_cache(image_paths, logger)
    if skipped_images:
        logger.info("Skipped %s unreadable images during feature cache build", len(skipped_images))
    prompt_texts = get_prompt_texts(args.prompt_set)
    clip_embeddings, embedded_paths = embed_images(
        image_paths=cached_paths,
        model_name=args.model,
        batch_size=max(1, args.batch_size),
        device=device,
        cache_dir=cache_dir,
        feature_cache=feature_cache,
        logger=logger,
    )

    visual_features, visual_paths = extract_visual_features(cached_paths, logger, feature_cache=feature_cache)
    item_features: np.ndarray | None = None
    if args.strict_same_corner_items:
        logger.info("Extracting CLIP prompt signatures for strict same-corner+items mode")
        item_features, item_paths = extract_clip_item_features(
            image_paths=embedded_paths,
            model_name=args.model,
            batch_size=max(1, args.batch_size),
            device=device,
            cache_dir=cache_dir,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
        )
        if embedded_paths != item_paths:
            path_set = set(embedded_paths) & set(item_paths)
            embedded_indices = [index for index, path in enumerate(embedded_paths) if path in path_set]
            item_indices = [index for index, path in enumerate(item_paths) if path in path_set]
            clip_embeddings = clip_embeddings[embedded_indices]
            embedded_paths = [embedded_paths[index] for index in embedded_indices]
            item_features = item_features[item_indices]

    if embedded_paths != visual_paths:
        path_set = set(embedded_paths) & set(visual_paths)
        embedded_indices = [index for index, path in enumerate(embedded_paths) if path in path_set]
        visual_indices = [index for index, path in enumerate(visual_paths) if path in path_set]
        clip_embeddings = clip_embeddings[embedded_indices]
        visual_features = visual_features[visual_indices]
        embedded_paths = [embedded_paths[index] for index in embedded_indices]
        if item_features is not None:
            item_features = item_features[embedded_indices]

    if args.local_descriptor_mode == "clip_tiles" and args.local_descriptor_weight > 0.0:
        logger.info(
            "Extracting learned local descriptors using CLIP tile crops (weight %.2f)",
            args.local_descriptor_weight,
        )
        populate_clip_tile_descriptors(
            image_paths=embedded_paths,
            model_name=args.model,
            device=device,
            cache_dir=cache_dir,
            feature_cache=feature_cache,
            logger=logger,
        )

    embeddings = combine_features(
        clip_embeddings=clip_embeddings,
        visual_features=visual_features,
        semantic_weight=args.semantic_weight,
        layout_weight=args.layout_weight,
        edge_weight=args.edge_weight,
        color_weight=args.color_weight,
    )
    logger.info(
        "Using feature weights semantic=%.2f layout=%.2f edge=%.2f color=%.2f",
        args.semantic_weight,
        args.layout_weight,
        args.edge_weight,
        args.color_weight,
    )
    logger.info(
        "Using preset=%s prompt_set=%s min_cluster_size=%s min_samples=%s cluster_epsilon=%.2f view_threshold=%.2f merge_threshold=%.2f merge_view_threshold=%.2f local_descriptor_mode=%s local_descriptor_weight=%.2f",
        args.preset,
        args.prompt_set,
        args.min_cluster_size,
        args.min_samples,
        args.cluster_epsilon,
        args.view_similarity_threshold,
        args.semantic_merge_threshold,
        args.merge_view_threshold,
        args.local_descriptor_mode,
        args.local_descriptor_weight,
    )

    logger.info("Clustering %s embedded images with two-stage HDBSCAN", len(embedded_paths))
    resolved_min_cluster_size = max(2, args.min_cluster_size)
    labels, merge_events = cluster_same_corner_groups(
        image_paths=embedded_paths,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=embeddings,
        item_features=item_features,
        feature_cache=feature_cache,
        min_cluster_size=resolved_min_cluster_size,
        min_samples=max(1, args.min_samples),
        cluster_epsilon=args.cluster_epsilon,
        view_max_cluster_size=args.view_max_cluster_size,
        view_similarity_threshold=args.view_similarity_threshold,
        semantic_merge_threshold=args.semantic_merge_threshold,
        merge_view_threshold=args.merge_view_threshold,
        strict_same_corner_items=args.strict_same_corner_items,
        item_similarity_threshold=args.item_similarity_threshold,
        strict_cluster_threshold=args.strict_cluster_threshold,
        semantic_similarity_floor=args.semantic_similarity_floor,
        view_linkage=args.view_linkage,
        strict_linkage=args.strict_linkage,
        orb_weight=args.orb_weight,
        structure_weight=args.structure_weight,
        local_descriptor_weight=args.local_descriptor_weight,
        logger=logger,
    )
    labels, noise_details, quality_reports, reassigned_images = finalize_noise_labels(
        image_paths=embedded_paths,
        labels=labels,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=embeddings,
        item_features=item_features,
        feature_cache=feature_cache,
        strict_same_corner_items=args.strict_same_corner_items,
        view_similarity_threshold=args.view_similarity_threshold,
        item_similarity_threshold=args.item_similarity_threshold,
        semantic_similarity_floor=args.semantic_similarity_floor,
        min_cluster_size=resolved_min_cluster_size,
        orb_weight=args.orb_weight,
        structure_weight=args.structure_weight,
        local_descriptor_weight=args.local_descriptor_weight,
        logger=logger,
    )

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    result = copy_clustered_images(
        embedded_paths,
        labels,
        output_dir,
        noise_details=noise_details,
        reassigned_images=reassigned_images,
        generate_contact_sheets=not args.skip_contact_sheets,
        feature_cache=feature_cache,
        merge_events=merge_events,
        skipped_images=skipped_images,
    )
    match_payload = write_match_scores(
        image_paths=embedded_paths,
        labels=labels,
        clip_embeddings=clip_embeddings,
        hybrid_embeddings=embeddings,
        item_features=item_features,
        feature_cache=feature_cache,
        strict_same_corner_items=args.strict_same_corner_items,
        view_similarity_threshold=args.view_similarity_threshold,
        item_similarity_threshold=args.item_similarity_threshold,
        semantic_similarity_floor=args.semantic_similarity_floor,
        orb_weight=args.orb_weight,
        structure_weight=args.structure_weight,
        local_descriptor_weight=args.local_descriptor_weight,
        quality_reports=quality_reports,
        noise_details=noise_details,
        reassigned_images=reassigned_images,
        output_dir=output_dir,
        logger=logger,
    )
    manifest = write_run_manifest(
        output_dir=output_dir,
        settings=resolved_settings,
        result=result,
        discovered_images=len(image_paths),
        processed_images=len(embedded_paths),
    )
    if not args.skip_html_summary:
        write_html_summary(output_dir=output_dir, result=result, manifest=manifest, match_payload=match_payload)

    logger.info("Clusters written to %s", output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
