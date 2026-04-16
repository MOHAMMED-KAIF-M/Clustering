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
SEGFORMER_ADE20K_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_FLAG_DETECTOR = "yolo_scene_clip"
DEFAULT_YOLO_MODEL = "yolov8n-seg.pt"
DEFAULT_YOLO_CONFIDENCE = 0.12
DEFAULT_YOLO_IOU = 0.45
DEFAULT_YOLO_IMAGE_SIZE = 1536
DEFAULT_YOLO_MAX_DETECTIONS = 100
DEFAULT_YOLO_RETINA_MASKS = True
DEFAULT_SCENE_CLIP_MIN_SCORE = 0.10
HYBRID_FALLBACK_MIN_LOCALIZED_OBJECT_FLAGS = 2
HYBRID_FALLBACK_SCORE_FLOOR = 0.05
DEFAULT_SAM_MODEL = "mobile_sam.pt"
DEFAULT_DEEPLAB_MODEL = "deeplabv3_resnet101"
DEFAULT_DEEPLAB_MIN_AREA_RATIO = 0.0025
DEFAULT_OPEN_VOCAB_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_OPEN_VOCAB_THRESHOLD = 0.10
DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_GROUNDING_DINO_THRESHOLD = 0.22
DEFAULT_GROUNDING_DINO_TEXT_THRESHOLD = 0.18
SEGFORMER_MAX_UPSAMPLED_SIDE = 1024

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
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
        "a real estate exterior photo of a building facade",
        "a real estate exterior photo of a yard",
        "the sky",
        "grass",
        "a tree",
        "plants",
        "a driveway",
        "a fence",
        "a patio",
        "a swimming pool",
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
    "visible_items": [
        "sky",
        "clouds",
        "outdoor",
        "grass",
        "lawn",
        "tree",
        "trees",
        "plants",
        "flowers",
        "building facade",
        "roof",
        "balcony railing",
        "patio",
        "driveway",
        "fence",
        "swimming pool",
        "window",
        "sliding glass door",
        "door",
        "curtains",
        "wall",
        "white wall",
        "floor",
        "floor tiles",
        "ceiling lights",
        "bed",
        "bedside table",
        "sofa",
        "television",
        "dining table",
        "chairs",
        "bench",
        "kitchen island",
        "kitchen cabinets",
        "countertop",
        "stove",
        "sink",
        "tap",
        "microwave",
        "refrigerator",
        "wardrobe",
        "toilet",
        "bathtub",
        "shower",
        "mirror",
    ],
}
YOLO_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed",),
    "bench": ("bench",),
    "sofa": ("couch", "sofa"),
    "television": ("tv",),
    "dining table": ("dining table",),
    "chairs": ("chair",),
    "stove": ("oven", "stove"),
    "sink": ("sink",),
    "microwave": ("microwave",),
    "refrigerator": ("refrigerator", "fridge"),
    "wardrobe": ("cabinet", "wardrobe", "closet"),
    "toilet": ("toilet",),
    "bathtub": ("bathtub",),
    "plants": ("potted plant", "plant", "plants"),
    "flowers": ("potted plant", "flower", "flowers"),
    "laptop": ("laptop",),
    "office chairs": ("chair",),
    "monitor": ("tv",),
    "cabinet": ("cabinet",),
}
SEGFORMER_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "clouds": ("sky",),
    "lawn": ("grass", "field"),
    "plants": ("plant",),
    "flowers": ("flower",),
    "building facade": ("building", "house"),
    "roof": ("house", "building"),
    "balcony railing": ("railing", "bannister"),
    "patio": ("path", "earth"),
    "driveway": ("road", "path", "dirt track"),
    "window": ("windowpane",),
    "sliding glass door": ("door", "windowpane"),
    "curtains": ("curtain",),
    "ceiling lights": ("light", "chandelier", "lamp"),
    "bedside table": ("table", "coffee table"),
    "television": ("crt screen", "computer"),
    "chairs": ("chair", "armchair"),
    "kitchen cabinets": ("cabinet", "buffet"),
    "wardrobe": ("cabinet", "chest of drawers"),
}
OPEN_VOCAB_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "bedside table": ("nightstand", "side table"),
    "television": ("tv", "television screen"),
    "kitchen island": ("island counter", "kitchen peninsula"),
    "kitchen cabinets": ("kitchen cabinet", "cabinets", "cabinet"),
    "countertop": ("counter top", "kitchen counter", "kitchen countertop"),
    "tap": ("tap", "faucet", "water tap", "sink faucet"),
    "microwave": ("microwave", "microwave oven"),
    "stove": ("oven", "cooktop", "range"),
    "refrigerator": ("fridge", "fridge freezer"),
    "wardrobe": ("closet", "cabinet"),
    "sliding glass door": ("sliding door", "glass door", "patio door"),
    "balcony railing": ("railing", "banister"),
    "swimming pool": ("pool",),
    "building facade": ("building exterior", "house exterior"),
    "ceiling lights": ("ceiling light", "light fixture"),
}
DEEPLAB_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "chairs": ("chair",),
    "dining table": ("diningtable",),
    "sofa": ("sofa",),
    "plants": ("pottedplant",),
    "flowers": ("pottedplant",),
    "television": ("tvmonitor",),
}
SCENE_FLAG_LABELS = {"wall", "white wall", "floor", "floor tiles", "sky", "clouds", "outdoor", "grass", "lawn", "tree", "trees"}
YOLO_SCENE_CLIP_OBJECT_LABELS = {
    "bed",
    "bench",
    "chairs",
    "sofa",
    "dining table",
    "microwave",
    "stove",
    "plants",
    "refrigerator",
    "sink",
    "toilet",
    "television",
}
YOLO_SCENE_CLIP_SCENE_LABELS = {
    "sky",
    "clouds",
    "outdoor",
    "grass",
    "lawn",
    "tree",
    "trees",
    "plants",
    "flowers",
    "building facade",
    "roof",
    "balcony railing",
    "patio",
    "driveway",
    "fence",
    "swimming pool",
    "window",
    "sliding glass door",
    "door",
    "curtains",
    "wall",
    "white wall",
    "floor",
    "floor tiles",
    "ceiling lights",
}
SCENE_LABEL_SCORE_FLOORS: dict[str, float] = {
    "sky": 0.16,
    "clouds": 0.16,
    "window": 0.16,
    "sliding glass door": 0.18,
    "white wall": 0.16,
    "wall": 0.18,
    "floor tiles": 0.16,
    "floor": 0.18,
    "tree": 0.18,
    "trees": 0.18,
    "grass": 0.18,
    "lawn": 0.18,
    "balcony railing": 0.18,
    "outdoor": 0.20,
    "patio": 0.20,
    "driveway": 0.20,
    "building facade": 0.22,
    "fence": 0.22,
    "swimming pool": 0.24,
}
LOCALIZED_FLAG_SOURCES = {"yolo", "open_vocab", "grounding_dino", "clip_tiles"}
HYBRID_CLIP_LABELS = {"outdoor"}
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
        help="Prompt set used for CLIP item signatures in strict same-corner+items mode.",
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
        "--flag-items",
        action="store_true",
        help="Generate per-image prompt-based item flags and include them in JSON/HTML outputs.",
    )
    parser.add_argument(
        "--flag-prompt-set",
        choices=sorted(PROMPT_SET_CONFIGS.keys()),
        default="visible_items",
        help="Prompt set used for item flags. This is separate from --prompt-set so flag labels can stay object-focused.",
    )
    parser.add_argument(
        "--flag-detector",
        choices=["yolo_scene_clip", "hybrid", "yolo", "open_vocab_hybrid", "open_vocab", "clip", "segformer_ade20k", "sam_deeplab_yolo_clip"],
        default=DEFAULT_FLAG_DETECTOR,
        help="Detection backend used for item flags. 'yolo_scene_clip' runs YOLOv8 segmentation for allowed object classes, then scores heuristic scene candidates with CLIP. 'hybrid' runs adaptive YOLOv8 segmentation first, then OWLv2 fallback on weak images, SegFormer semantic segmentation, and CLIP last-resort prompts. 'sam_deeplab_yolo_clip' uses YOLOv8 + DeepLabV3 + CLIP, then refines regions with SAM. 'open_vocab_hybrid' uses OWLv2 + SegFormer + CLIP. 'open_vocab' uses OWLv2 only, 'yolo' uses YOLOv8 segmentation only, 'clip' uses prompt similarity, and 'segformer_ade20k' uses semantic segmentation.",
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help="YOLO weights used when --flag-items is enabled.",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=DEFAULT_YOLO_CONFIDENCE,
        help="Minimum YOLO detection confidence used when --flag-items is enabled.",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=DEFAULT_YOLO_IOU,
        help="YOLO NMS IoU threshold used when --flag-items is enabled.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=DEFAULT_YOLO_IMAGE_SIZE,
        help="YOLO inference image size used when --flag-items is enabled.",
    )
    parser.add_argument(
        "--yolo-max-det",
        type=int,
        default=DEFAULT_YOLO_MAX_DETECTIONS,
        help="Maximum number of YOLO detections retained per image or tile view.",
    )
    parser.add_argument(
        "--yolo-retina-masks",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_YOLO_RETINA_MASKS,
        help="Request full-resolution segmentation masks from YOLO when supported.",
    )
    parser.add_argument(
        "--open-vocab-model",
        default=DEFAULT_OPEN_VOCAB_MODEL,
        help="Open-vocabulary detection model used when --flag-detector hybrid, open_vocab_hybrid, or open_vocab is enabled.",
    )
    parser.add_argument(
        "--open-vocab-threshold",
        type=float,
        default=DEFAULT_OPEN_VOCAB_THRESHOLD,
        help="Minimum open-vocabulary detection confidence used when --flag-detector hybrid, open_vocab_hybrid, or open_vocab is enabled.",
    )
    parser.add_argument(
        "--flag-top-k",
        type=int,
        default=16,
        help="Maximum number of item flags to keep per image when --flag-items is enabled.",
    )
    parser.add_argument(
        "--flag-min-score",
        type=float,
        default=0.05,
        help="Minimum prompt score required before an item label is surfaced when --flag-items is enabled.",
    )
    parser.add_argument(
        "--scene-clip-min-score",
        type=float,
        default=DEFAULT_SCENE_CLIP_MIN_SCORE,
        help="Minimum CLIP score required before a heuristic scene-region label is kept when --flag-detector yolo_scene_clip is enabled.",
    )
    parser.add_argument(
        "--flag-include-labels",
        nargs="+",
        default=None,
        help="Optional list of normalized flag labels to keep, for example: --flag-include-labels sky grass",
    )
    parser.add_argument(
        "--segmentation-min-area",
        type=float,
        default=0.0025,
        help="Minimum image-area ratio required before a segmentation label is kept when --flag-detector hybrid, open_vocab_hybrid, or segformer_ade20k is enabled.",
    )
    parser.add_argument(
        "--sam-model",
        default=DEFAULT_SAM_MODEL,
        help="SAM weights used when --flag-detector sam_deeplab_yolo_clip is enabled.",
    )
    parser.add_argument(
        "--deeplab-model",
        choices=["deeplabv3_resnet50", "deeplabv3_resnet101"],
        default=DEFAULT_DEEPLAB_MODEL,
        help="DeepLabV3 backbone used when --flag-detector sam_deeplab_yolo_clip is enabled.",
    )
    parser.add_argument(
        "--deeplab-min-area",
        type=float,
        default=DEFAULT_DEEPLAB_MIN_AREA_RATIO,
        help="Minimum image-area ratio required before a DeepLabV3 label is kept when --flag-detector sam_deeplab_yolo_clip is enabled.",
    )
    parser.add_argument(
        "--annotate-flagged-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Draw detected object boundaries and labels onto the copied output images. Defaults to enabled when --flag-items is used.",
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


def deduplicate_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    unique_boxes: list[tuple[int, int, int, int]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for box in boxes:
        if box in seen_boxes:
            continue
        seen_boxes.add(box)
        unique_boxes.append(box)
    return unique_boxes


def generate_clip_tile_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
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

    normalized_boxes: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom in candidate_boxes:
        normalized_left = max(0, min(left, width - crop_width))
        normalized_top = max(0, min(top, height - crop_height))
        normalized_boxes.append(
            (
                normalized_left,
                normalized_top,
                normalized_left + crop_width,
                normalized_top + crop_height,
            )
        )
    return deduplicate_boxes(normalized_boxes)


def generate_flag_region_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    boxes: list[tuple[int, int, int, int]] = []
    for width_ratio, height_ratio in ((0.72, 0.72), (0.52, 0.52)):
        crop_width = min(width, max(64, int(width * width_ratio)))
        crop_height = min(height, max(64, int(height * height_ratio)))
        x_positions = sorted({0, max(0, (width - crop_width) // 2), max(0, width - crop_width)})
        y_positions = sorted({0, max(0, (height - crop_height) // 2), max(0, height - crop_height)})
        for top in y_positions:
            for left in x_positions:
                boxes.append((left, top, left + crop_width, top + crop_height))
    return deduplicate_boxes(boxes)


def generate_detector_tile_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    boxes: list[tuple[int, int, int, int]] = []
    for width_ratio, height_ratio in ((0.78, 0.78), (0.58, 0.58)):
        crop_width = min(width, max(96, int(width * width_ratio)))
        crop_height = min(height, max(96, int(height * height_ratio)))
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
        for left, top, right, bottom in candidate_boxes:
            normalized_left = max(0, min(left, width - crop_width))
            normalized_top = max(0, min(top, height - crop_height))
            boxes.append(
                (
                    normalized_left,
                    normalized_top,
                    normalized_left + crop_width,
                    normalized_top + crop_height,
                )
            )
    full_image_box = (0, 0, width, height)
    return [box for box in deduplicate_boxes(boxes) if box != full_image_box]


def generate_flag_detection_views(image: Image.Image) -> list[tuple[tuple[int, int, int, int], Image.Image]]:
    full_image_box = (0, 0, image.width, image.height)
    views: list[tuple[tuple[int, int, int, int], Image.Image]] = [(full_image_box, image)]
    for box in generate_detector_tile_boxes(image):
        views.append((box, image.crop(box)))
    return views


def remap_box_to_image(
    left: int,
    top: int,
    right: int,
    bottom: int,
    view_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    view_left, view_top, _, _ = view_box
    mapped_left = view_left + int(left)
    mapped_top = view_top + int(top)
    mapped_right = view_left + int(right)
    mapped_bottom = view_top + int(bottom)
    mapped_left = max(0, min(mapped_left, max(0, image_width - 1)))
    mapped_top = max(0, min(mapped_top, max(0, image_height - 1)))
    mapped_right = max(mapped_left + 1, min(mapped_right, image_width))
    mapped_bottom = max(mapped_top + 1, min(mapped_bottom, image_height))
    return mapped_left, mapped_top, mapped_right, mapped_bottom


def remap_boundary_to_image(
    boundary: list[dict[str, int]] | None,
    *,
    view_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[dict[str, int]] | None:
    if not boundary:
        return None

    view_left, view_top, _, _ = view_box
    remapped_boundary: list[dict[str, int]] = []
    seen_points: set[tuple[int, int]] = set()
    for point in boundary:
        if not isinstance(point, dict):
            continue
        try:
            x = view_left + int(round(float(point.get("x", 0))))
            y = view_top + int(round(float(point.get("y", 0))))
        except (TypeError, ValueError):
            continue
        x = max(0, min(x, max(0, image_width - 1)))
        y = max(0, min(y, max(0, image_height - 1)))
        key = (x, y)
        if key in seen_points:
            continue
        seen_points.add(key)
        remapped_boundary.append({"x": x, "y": y})
    return remapped_boundary if len(remapped_boundary) >= 3 else None


def boundary_region_from_points(
    boundary: list[dict[str, int]] | None,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, int] | None:
    if not boundary or image_width <= 0 or image_height <= 0:
        return None

    try:
        xs = [int(point["x"]) for point in boundary]
        ys = [int(point["y"]) for point in boundary]
    except (KeyError, TypeError, ValueError):
        return None
    if not xs or not ys:
        return None

    left = max(0, min(xs))
    top = max(0, min(ys))
    right = max(left + 1, min(max(xs) + 1, image_width))
    bottom = max(top + 1, min(max(ys) + 1, image_height))
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def polygon_to_boundary_points(
    polygon: object,
    *,
    image_width: int,
    image_height: int,
) -> list[dict[str, int]] | None:
    try:
        contour = np.asarray(polygon, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] < 2:
        return None

    contour = contour[:, :2].reshape(-1, 1, 2)
    epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    if simplified is None or simplified.shape[0] < 3:
        simplified = contour

    boundary: list[dict[str, int]] = []
    seen_points: set[tuple[int, int]] = set()
    for point in simplified.reshape(-1, 2):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = max(0, min(x, max(0, image_width - 1)))
        y = max(0, min(y, max(0, image_height - 1)))
        key = (x, y)
        if key in seen_points:
            continue
        seen_points.add(key)
        boundary.append({"x": x, "y": y})
    return boundary if len(boundary) >= 3 else None


def extract_yolo_mask_region(
    result: object,
    detection_index: int,
    *,
    view_width: int,
    view_height: int,
) -> tuple[dict[str, int] | None, list[dict[str, int]] | None]:
    masks = getattr(result, "masks", None) if result is not None else None
    if masks is None:
        return None, None

    polygons = getattr(masks, "xy", None)
    if isinstance(polygons, (list, tuple)) and 0 <= detection_index < len(polygons):
        polygon_boundary = polygon_to_boundary_points(
            polygons[detection_index],
            image_width=view_width,
            image_height=view_height,
        )
        polygon_region = boundary_region_from_points(
            polygon_boundary,
            image_width=view_width,
            image_height=view_height,
        )
        if polygon_region is not None and polygon_boundary is not None:
            return polygon_region, polygon_boundary

    mask_data = getattr(masks, "data", None)
    if mask_data is None:
        return None, None

    try:
        mask_slice = mask_data[detection_index]
    except (IndexError, TypeError):
        return None, None

    if hasattr(mask_slice, "detach"):
        mask_array = mask_slice.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_slice)
    mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        return None, None

    component_region, component_mask, _ = largest_component_region(mask_array > 0.5)
    if component_region is None or component_mask is None:
        return None, None

    boundary = mask_to_boundary_points(component_mask)
    if not boundary:
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned_mask = cv2.morphologyEx(component_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        component_region, component_mask, _ = largest_component_region(cleaned_mask.astype(bool))
        if component_region is None or component_mask is None:
            return None, None
        boundary = mask_to_boundary_points(component_mask)
    if not boundary:
        return None, None

    mask_height, mask_width = component_mask.shape
    scaled_region = scale_region_to_image(
        region=component_region,
        source_width=mask_width,
        source_height=mask_height,
        target_width=view_width,
        target_height=view_height,
    )
    scaled_boundary = scale_boundary_to_image(
        boundary=boundary,
        source_width=mask_width,
        source_height=mask_height,
        target_width=view_width,
        target_height=view_height,
    )
    if boundary_region_from_points(
        scaled_boundary,
        image_width=view_width,
        image_height=view_height,
    ) is None:
        scaled_boundary = None
    return scaled_region, scaled_boundary


def generate_clip_tile_crops(image: Image.Image) -> list[Image.Image]:
    return [image.crop(box) for box in generate_clip_tile_boxes(image)]


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
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []

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
    prompt_score_batches: list[np.ndarray] = []
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
            prompt_score_batches.append(prompt_distribution.cpu().numpy().astype(np.float32, copy=False))
            valid_paths.extend(batch_valid_paths)

    if not prompt_score_batches:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []

    prompt_scores = np.concatenate(prompt_score_batches, axis=0).astype(np.float32, copy=False)
    return l2_normalize(prompt_scores), prompt_scores, valid_paths


def extract_clip_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

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

    combined_scores: list[np.ndarray] = []
    combined_regions: list[list[dict | None]] = []
    valid_paths: list[Path] = []
    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            cached = feature_cache.get(image_path) if feature_cache is not None else None
            image = cached.image if cached is not None else load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during item flag extraction: %s", image_path)
                continue

            tile_boxes = generate_flag_region_boxes(image)
            views = [image] + [image.crop(box) for box in tile_boxes]
            view_tensor = torch.stack([preprocess(view) for view in views]).to(device)
            view_features = model.encode_image(view_tensor).float()
            view_features = view_features / view_features.norm(dim=-1, keepdim=True)
            similarity = view_features @ prompt_features.T
            prompt_distribution = torch.softmax(similarity * 10.0, dim=1).cpu().numpy().astype(np.float32, copy=False)
            combined_scores.append(np.clip(prompt_distribution.max(axis=0), 0.0, 1.0))
            if tile_boxes:
                tile_distribution = prompt_distribution[1:]
                best_tile_indices = np.argmax(tile_distribution, axis=0)
                image_regions: list[dict | None] = []
                for prompt_index, tile_index in enumerate(best_tile_indices):
                    left, top, right, bottom = tile_boxes[int(tile_index)]
                    image_regions.append(
                        build_region_payload(
                            int(left),
                            int(top),
                            int(right),
                            int(bottom),
                            boundary=box_to_boundary_points(int(left), int(top), int(right), int(bottom)),
                            extras={
                                "prompt_region_score_percent": similarity_to_percent(
                                float(tile_distribution[int(tile_index), int(prompt_index)])
                                ),
                                "source": "clip_tiles",
                            },
                        )
                    )
            else:
                image_regions = [None for _ in prompt_texts]
            combined_regions.append(image_regions)
            valid_paths.append(image_path)

            if index % 8 == 0 or index == len(image_paths):
                logger.info("Scored crop-aware item flags for %s/%s images", index, len(image_paths))

    if not combined_scores:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.stack(combined_scores).astype(np.float32, copy=False), combined_regions, valid_paths


def load_yolo_runtime(model_path: str) -> object:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The ultralytics package is required for --flag-detector yolo. "
            "Install the updated requirements, then rerun."
        ) from exc

    try:
        return YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load YOLO weights '{model_path}'. Provide a valid local path or allow the weights to download, then rerun."
        ) from exc


def load_sam_runtime(model_path: str) -> object:
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise RuntimeError(
            "The ultralytics package with SAM support is required for --flag-detector sam_deeplab_yolo_clip. "
            "Install the updated requirements, then rerun."
        ) from exc

    try:
        return SAM(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load SAM weights '{model_path}'. Provide a valid local path or allow the weights to download, then rerun."
        ) from exc


def load_open_vocab_runtime(model_id: str, device: str) -> tuple[object, object]:
    try:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError as exc:
        raise RuntimeError(
            "The transformers package with OWLv2 support is required for --flag-detector open_vocab "
            "or open_vocab_hybrid or advanced_hybrid. Install the updated requirements, then rerun."
        ) from exc

    try:
        processor = Owlv2Processor.from_pretrained(model_id, local_files_only=True, use_fast=True)
        model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=True)
    except OSError:
        try:
            processor = Owlv2Processor.from_pretrained(model_id, use_fast=True)
            model = Owlv2ForObjectDetection.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the open-vocabulary model "
                f"'{model_id}'. If it is not cached locally, allow it to download and then rerun."
            ) from exc

    model.to(device)
    model.eval()
    return processor, model


def load_grounding_dino_runtime(model_id: str, device: str) -> tuple[object, object]:
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "The transformers package with Grounding DINO support is required for --flag-detector grounding_dino "
            "or grounding_dino_hybrid or advanced_hybrid. Install the updated requirements, then rerun."
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True, use_fast=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True)
    except OSError:
        try:
            processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the Grounding DINO model "
                f"'{model_id}'. If it is not cached locally, allow it to download and then rerun."
            ) from exc

    model.to(device)
    model.eval()
    return processor, model


def extract_grounding_dino_flag_scores(
    image_paths: list[Path],
    device: str,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    model_id: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths or not prompt_texts:
        return np.empty((0, 0), dtype=np.float32), [], []

    expanded_prompt_texts, expanded_prompt_targets, prompt_label_to_indices = build_grounding_dino_prompt_variants(prompt_texts)
    if not expanded_prompt_texts:
        return np.empty((0, 0), dtype=np.float32), [], []

    processor, model = load_grounding_dino_runtime(model_id=model_id, device=device)
    runtime_device = device
    box_threshold = float(np.clip(box_threshold, 0.0, 1.0))
    text_threshold = float(np.clip(text_threshold, 0.0, 1.0))

    combined_scores: list[np.ndarray] = []
    combined_regions: list[list[dict | None]] = []
    valid_paths: list[Path] = []

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            cached = feature_cache.get(image_path) if feature_cache is not None else None
            image = cached.image if cached is not None else load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during Grounding DINO detection: %s", image_path)
                continue

            image_scores = np.zeros(len(prompt_texts), dtype=np.float32)
            image_regions: list[dict | None] = [None for _ in prompt_texts]
            detection_views = generate_flag_detection_views(image)

            for view_box, detection_view in detection_views:
                encoded = processor(images=detection_view, text=[expanded_prompt_texts], return_tensors="pt")
                encoded = {
                    key: value.to(runtime_device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }

                try:
                    outputs = model(**encoded)
                except torch.OutOfMemoryError:
                    if runtime_device != "cpu":
                        logger.warning(
                            "Grounding DINO ran out of GPU memory on %s. Retrying Grounding DINO on CPU for the remaining images.",
                            image_path.name,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        processor, model = load_grounding_dino_runtime(model_id=model_id, device="cpu")
                        runtime_device = "cpu"
                        encoded = processor(images=detection_view, text=[expanded_prompt_texts], return_tensors="pt")
                        encoded = {
                            key: value.to(runtime_device) if hasattr(value, "to") else value
                            for key, value in encoded.items()
                        }
                        outputs = model(**encoded)
                    else:
                        raise

                input_ids = encoded.get("input_ids")
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(detection_view.height, detection_view.width)],
                    text_labels=[expanded_prompt_texts],
                )
                result = results[0] if results else {}
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                labels = result.get("labels", [])

                detection_count = min(len(boxes), len(scores))
                for detection_index in range(detection_count):
                    try:
                        score = float(np.clip(float(scores[detection_index]), 0.0, 1.0))
                        left, top, right, bottom = [int(round(float(value))) for value in boxes[detection_index].tolist()]
                    except (AttributeError, IndexError, TypeError, ValueError):
                        continue

                    matched_label = ""
                    prompt_indices: list[int] = []
                    if detection_index < len(labels):
                        raw_label = labels[detection_index]
                        if isinstance(raw_label, (list, tuple)):
                            label_candidates = [normalize_detector_label(str(item)) for item in raw_label]
                            matched_label = ", ".join(str(item) for item in raw_label)
                        else:
                            label_candidates = [normalize_detector_label(str(raw_label))]
                            matched_label = str(raw_label)
                        for candidate_label in label_candidates:
                            if not candidate_label:
                                continue
                            prompt_indices.extend(prompt_label_to_indices.get(candidate_label, []))
                            stripped_candidate = normalize_detector_label(
                                prompt_text_to_label(candidate_label) or candidate_label
                            )
                            if stripped_candidate and stripped_candidate != candidate_label:
                                prompt_indices.extend(prompt_label_to_indices.get(stripped_candidate, []))

                    if not prompt_indices:
                        continue

                    prompt_indices = sorted({int(prompt_index) for prompt_index in prompt_indices})
                    left = max(0, min(left, max(0, detection_view.width - 1)))
                    top = max(0, min(top, max(0, detection_view.height - 1)))
                    right = max(left + 1, min(right, detection_view.width))
                    bottom = max(top + 1, min(bottom, detection_view.height))
                    left, top, right, bottom = remap_box_to_image(
                        left,
                        top,
                        right,
                        bottom,
                        view_box=view_box,
                        image_width=image.width,
                        image_height=image.height,
                    )
                    region_payload = build_region_payload(
                        left,
                        top,
                        right,
                        bottom,
                        boundary=box_to_boundary_points(left, top, right, bottom),
                        extras={
                            "source": "grounding_dino",
                            "grounding_dino_model": model_id,
                            "grounding_dino_label": matched_label,
                            "view": "tile" if view_box != (0, 0, image.width, image.height) else "full_image",
                        },
                    )
                    for prompt_index in prompt_indices:
                        if prompt_index < 0 or prompt_index >= len(prompt_texts):
                            continue
                        if score <= float(image_scores[prompt_index]):
                            continue
                        image_scores[prompt_index] = score
                        image_regions[prompt_index] = dict(region_payload)

            combined_scores.append(image_scores)
            combined_regions.append(image_regions)
            valid_paths.append(image_path)

            if index % 4 == 0 or index == len(image_paths):
                logger.info("Scored Grounding DINO detections for %s/%s images", index, len(image_paths))

    if not combined_scores:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.stack(combined_scores).astype(np.float32, copy=False), combined_regions, valid_paths


def extract_open_vocab_flag_scores(
    image_paths: list[Path],
    device: str,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    model_id: str,
    score_threshold: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths or not prompt_texts:
        return np.empty((0, 0), dtype=np.float32), [], []

    expanded_prompt_texts, expanded_prompt_targets, prompt_label_to_indices = build_open_vocab_prompt_variants(prompt_texts)
    if not expanded_prompt_texts:
        return np.empty((0, 0), dtype=np.float32), [], []

    processor, model = load_open_vocab_runtime(model_id=model_id, device=device)
    runtime_device = device
    score_threshold = float(np.clip(score_threshold, 0.0, 1.0))

    combined_scores: list[np.ndarray] = []
    combined_regions: list[list[dict | None]] = []
    valid_paths: list[Path] = []

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            cached = feature_cache.get(image_path) if feature_cache is not None else None
            image = cached.image if cached is not None else load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during open-vocabulary detection: %s", image_path)
                continue

            image_scores = np.zeros(len(prompt_texts), dtype=np.float32)
            image_regions: list[dict | None] = [None for _ in prompt_texts]
            detection_views = generate_flag_detection_views(image)
            for view_box, detection_view in detection_views:
                encoded = processor(images=detection_view, text=[expanded_prompt_texts], return_tensors="pt")
                encoded = {
                    key: value.to(runtime_device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }

                try:
                    outputs = model(**encoded)
                except torch.OutOfMemoryError:
                    if runtime_device != "cpu":
                        logger.warning(
                            "OWLv2 ran out of GPU memory on %s. Retrying open-vocabulary detection on CPU for the remaining images.",
                            image_path.name,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        processor, model = load_open_vocab_runtime(model_id=model_id, device="cpu")
                        runtime_device = "cpu"
                        encoded = processor(images=detection_view, text=[expanded_prompt_texts], return_tensors="pt")
                        encoded = {
                            key: value.to(runtime_device) if hasattr(value, "to") else value
                            for key, value in encoded.items()
                        }
                        outputs = model(**encoded)
                    else:
                        raise

                results = processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=[(detection_view.height, detection_view.width)],
                    threshold=score_threshold,
                    text_labels=[expanded_prompt_texts],
                )
                result = results[0] if results else {}
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                labels = result.get("labels")
                text_labels = result.get("text_labels")

                detection_count = min(len(boxes), len(scores))
                for detection_index in range(detection_count):
                    try:
                        score = float(np.clip(float(scores[detection_index]), 0.0, 1.0))
                        left, top, right, bottom = [int(round(float(value))) for value in boxes[detection_index].tolist()]
                    except (AttributeError, IndexError, TypeError, ValueError):
                        continue

                    prompt_indices: list[int] = []
                    matched_label = None
                    if labels is not None and detection_index < len(labels):
                        try:
                            expanded_prompt_index = int(labels[detection_index])
                        except (TypeError, ValueError):
                            expanded_prompt_index = -1
                        if 0 <= expanded_prompt_index < len(expanded_prompt_targets):
                            prompt_indices = [expanded_prompt_targets[expanded_prompt_index]]
                            matched_label = expanded_prompt_texts[expanded_prompt_index]

                    if not prompt_indices and text_labels is not None and detection_index < len(text_labels):
                        matched_label = prompt_text_to_label(str(text_labels[detection_index])) or str(text_labels[detection_index])
                        prompt_indices = prompt_label_to_indices.get(normalize_detector_label(matched_label), [])

                    if not prompt_indices:
                        continue

                    left = max(0, min(left, max(0, detection_view.width - 1)))
                    top = max(0, min(top, max(0, detection_view.height - 1)))
                    right = max(left + 1, min(right, detection_view.width))
                    bottom = max(top + 1, min(bottom, detection_view.height))
                    left, top, right, bottom = remap_box_to_image(
                        left,
                        top,
                        right,
                        bottom,
                        view_box=view_box,
                        image_width=image.width,
                        image_height=image.height,
                    )
                    region_payload = build_region_payload(
                        left,
                        top,
                        right,
                        bottom,
                        boundary=box_to_boundary_points(left, top, right, bottom),
                        extras={
                            "source": "open_vocab",
                            "open_vocab_model": model_id,
                            "open_vocab_label": matched_label or (
                                prompt_texts[prompt_indices[0]]
                                if prompt_indices and 0 <= prompt_indices[0] < len(prompt_texts)
                                else ""
                            ),
                            "view": "tile" if view_box != (0, 0, image.width, image.height) else "full_image",
                        },
                    )
                    for prompt_index in prompt_indices:
                        if prompt_index < 0 or prompt_index >= len(prompt_texts):
                            continue
                        if score <= float(image_scores[prompt_index]):
                            continue
                        image_scores[prompt_index] = score
                        image_regions[prompt_index] = dict(region_payload)

            combined_scores.append(image_scores)
            combined_regions.append(image_regions)
            valid_paths.append(image_path)

            if index % 4 == 0 or index == len(image_paths):
                logger.info("Scored open-vocabulary detections for %s/%s images", index, len(image_paths))

    if not combined_scores:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.stack(combined_scores).astype(np.float32, copy=False), combined_regions, valid_paths


def extract_yolo_flag_scores(
    image_paths: list[Path],
    device: str,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    model_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    max_detections: int,
    retina_masks: bool,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    model = load_yolo_runtime(model_path=model_path)
    class_names = resolve_named_class_lookup(
        getattr(model, "names", None) or getattr(getattr(model, "model", None), "names", None)
    )
    prompt_class_lookup = build_yolo_prompt_class_lookup(prompt_texts, class_names)
    if not prompt_class_lookup:
        logger.warning("None of the selected prompt labels map to classes in YOLO model %s", model_path)

    predict_device = "0" if device == "cuda" else "cpu"
    confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
    iou_threshold = float(np.clip(iou_threshold, 0.0, 1.0))
    image_size = max(32, int(image_size))
    max_detections = max(1, int(max_detections))
    retina_masks = bool(retina_masks)

    combined_scores: list[np.ndarray] = []
    combined_regions: list[list[dict | None]] = []
    valid_paths: list[Path] = []

    for index, image_path in enumerate(image_paths, start=1):
        cached = feature_cache.get(image_path) if feature_cache is not None else None
        image = cached.image if cached is not None else load_rgb_image(image_path)
        if image is None:
            logger.warning("Skipping unreadable image during YOLO item detection: %s", image_path)
            continue

        image_scores = np.zeros(len(prompt_texts), dtype=np.float32)
        image_regions: list[dict | None] = [None for _ in prompt_texts]
        detection_views = generate_flag_detection_views(image)
        for view_box, detection_view in detection_views:
            results = model.predict(
                source=detection_view,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                retina_masks=retina_masks,
                max_det=max_detections,
                device=predict_device,
                verbose=False,
            )
            result = results[0] if results else None
            boxes = getattr(result, "boxes", None) if result is not None else None

            if boxes is None or len(boxes) == 0:
                continue

            for detection_index in range(len(boxes)):
                try:
                    class_id = int(boxes.cls[detection_index].item())
                    score = float(np.clip(boxes.conf[detection_index].item(), 0.0, 1.0))
                    left, top, right, bottom = [int(round(float(value))) for value in boxes.xyxy[detection_index].tolist()]
                except (AttributeError, IndexError, TypeError, ValueError):
                    continue

                prompt_indices = prompt_class_lookup.get(class_id)
                if not prompt_indices:
                    continue

                boundary_source = "box"
                mask_region, mask_boundary = extract_yolo_mask_region(
                    result,
                    detection_index,
                    view_width=detection_view.width,
                    view_height=detection_view.height,
                )

                left = max(0, min(left, max(0, detection_view.width - 1)))
                top = max(0, min(top, max(0, detection_view.height - 1)))
                right = max(left + 1, min(right, detection_view.width))
                bottom = max(top + 1, min(bottom, detection_view.height))
                left, top, right, bottom = remap_box_to_image(
                    mask_region["left"] if mask_region is not None else left,
                    mask_region["top"] if mask_region is not None else top,
                    mask_region["right"] if mask_region is not None else right,
                    mask_region["bottom"] if mask_region is not None else bottom,
                    view_box=view_box,
                    image_width=image.width,
                    image_height=image.height,
                )
                boundary = box_to_boundary_points(left, top, right, bottom)
                if mask_boundary:
                    remapped_boundary = remap_boundary_to_image(
                        mask_boundary,
                        view_box=view_box,
                        image_width=image.width,
                        image_height=image.height,
                    )
                    remapped_region = boundary_region_from_points(
                        remapped_boundary,
                        image_width=image.width,
                        image_height=image.height,
                    )
                    if remapped_region is not None and remapped_boundary is not None:
                        left = int(remapped_region["left"])
                        top = int(remapped_region["top"])
                        right = int(remapped_region["right"])
                        bottom = int(remapped_region["bottom"])
                        boundary = remapped_boundary
                        boundary_source = "mask"
                region_payload = build_region_payload(
                    left,
                    top,
                    right,
                    bottom,
                    boundary=boundary,
                    extras={
                        "source": "yolo",
                        "boundary_source": boundary_source,
                        "yolo_class": class_names.get(class_id, str(class_id)),
                        "view": "tile" if view_box != (0, 0, image.width, image.height) else "full_image",
                    },
                )
                for prompt_index in prompt_indices:
                    if score <= float(image_scores[prompt_index]):
                        continue
                    image_scores[prompt_index] = score
                    image_regions[prompt_index] = dict(region_payload)

        combined_scores.append(image_scores)
        combined_regions.append(image_regions)
        valid_paths.append(image_path)

        if index % 8 == 0 or index == len(image_paths):
            logger.info("Scored YOLO item detections for %s/%s images", index, len(image_paths))

    if not combined_scores:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.stack(combined_scores).astype(np.float32, copy=False), combined_regions, valid_paths


def resolve_deeplabv3_class_names(model_name: str) -> dict[int, str]:
    try:
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            DeepLabV3_ResNet101_Weights,
        )
    except ImportError as exc:
        raise RuntimeError(
            "torchvision with DeepLabV3 support is required for --flag-detector sam_deeplab_yolo_clip. "
            "Install the updated requirements, then rerun."
        ) from exc

    model_name = str(model_name).strip().lower()
    if model_name == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
    elif model_name == "deeplabv3_resnet101":
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
    else:
        raise RuntimeError(f"Unsupported DeepLabV3 model '{model_name}'.")
    return resolve_named_class_lookup(weights.meta.get("categories", []))


def load_deeplabv3_runtime(model_name: str, device: str) -> tuple[object, object, dict[int, str]]:
    try:
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            DeepLabV3_ResNet101_Weights,
            deeplabv3_resnet50,
            deeplabv3_resnet101,
        )
    except ImportError as exc:
        raise RuntimeError(
            "torchvision with DeepLabV3 support is required for --flag-detector sam_deeplab_yolo_clip. "
            "Install the updated requirements, then rerun."
        ) from exc

    model_name = str(model_name).strip().lower()
    if model_name == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        build_model = deeplabv3_resnet50
    elif model_name == "deeplabv3_resnet101":
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        build_model = deeplabv3_resnet101
    else:
        raise RuntimeError(f"Unsupported DeepLabV3 model '{model_name}'.")

    try:
        model = build_model(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load DeepLabV3 model '{model_name}'. If weights are not cached locally, allow them to download and then rerun."
        ) from exc

    model.to(device)
    model.eval()
    preprocess = weights.transforms()
    class_names = resolve_deeplabv3_class_names(model_name)
    return preprocess, model, class_names


def run_deeplabv3_inference(
    preprocess: object,
    model: object,
    image: Image.Image,
    device: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    logits = model(image_tensor)["out"][0]
    probabilities = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    height = int(probabilities.shape[1])
    width = int(probabilities.shape[2])
    return probabilities, (width, height)


def load_segformer_ade20k_runtime(device: str) -> tuple[object, object, dict[str, int]]:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    try:
        processor = AutoImageProcessor.from_pretrained(
            SEGFORMER_ADE20K_MODEL_ID,
            local_files_only=True,
            use_fast=True,
        )
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ADE20K_MODEL_ID, local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            "The semantic segmentation model is not cached locally. Download "
            f"{SEGFORMER_ADE20K_MODEL_ID} first, then rerun with --flag-detector segformer_ade20k."
        ) from exc
    model.to(device)
    model.eval()
    label_to_id = {
        prompt_text_to_label(str(label)).lower(): int(label_id)
        for label_id, label in model.config.id2label.items()
        if prompt_text_to_label(str(label))
    }
    return processor, model, label_to_id


def run_segformer_ade20k_inference(
    processor: object,
    model: object,
    image: Image.Image,
    device: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    target_width = int(image.width)
    target_height = int(image.height)
    max_side = max(target_width, target_height)
    if max_side > SEGFORMER_MAX_UPSAMPLED_SIDE:
        scale = float(SEGFORMER_MAX_UPSAMPLED_SIDE / max_side)
        target_width = max(1, int(round(target_width * scale)))
        target_height = max(1, int(round(target_height * scale)))

    encoded = processor(images=image, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded).logits
    upsampled_logits = F.interpolate(
        logits,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return (
        upsampled_logits.softmax(dim=1)[0].detach().cpu().numpy().astype(np.float32, copy=False),
        (target_width, target_height),
    )


def scale_region_to_image(
    region: dict[str, int],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> dict[str, int]:
    if source_width <= 0 or source_height <= 0:
        return dict(region)

    scale_x = float(target_width / source_width)
    scale_y = float(target_height / source_height)
    left = int(round(float(region["left"]) * scale_x))
    top = int(round(float(region["top"]) * scale_y))
    right = int(round(float(region["right"]) * scale_x))
    bottom = int(round(float(region["bottom"]) * scale_y))
    left = max(0, min(left, max(0, target_width - 1)))
    top = max(0, min(top, max(0, target_height - 1)))
    right = max(left + 1, min(right, target_width))
    bottom = max(top + 1, min(bottom, target_height))
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def scale_boundary_to_image(
    boundary: list[dict[str, int]] | None,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> list[dict[str, int]] | None:
    if not boundary or source_width <= 0 or source_height <= 0:
        return boundary

    scale_x = float(target_width / source_width)
    scale_y = float(target_height / source_height)
    scaled_boundary: list[dict[str, int]] = []
    seen_points: set[tuple[int, int]] = set()
    for point in boundary:
        x = int(round(float(point["x"]) * scale_x))
        y = int(round(float(point["y"]) * scale_y))
        x = max(0, min(x, max(0, target_width - 1)))
        y = max(0, min(y, max(0, target_height - 1)))
        key = (x, y)
        if key in seen_points:
            continue
        seen_points.add(key)
        scaled_boundary.append({"x": x, "y": y})

    return scaled_boundary if len(scaled_boundary) >= 3 else boundary


def largest_component_region(mask: np.ndarray) -> tuple[dict | None, np.ndarray | None, int]:
    mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
    if mask_uint8.ndim != 2 or not mask_uint8.any():
        return None, None, 0

    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if component_count <= 1:
        return None, None, 0

    best_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[best_index, cv2.CC_STAT_AREA])
    if area <= 0:
        return None, None, 0

    left = int(stats[best_index, cv2.CC_STAT_LEFT])
    top = int(stats[best_index, cv2.CC_STAT_TOP])
    width = int(stats[best_index, cv2.CC_STAT_WIDTH])
    height = int(stats[best_index, cv2.CC_STAT_HEIGHT])
    component_mask = component_labels == best_index
    return (
        {
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
        },
        component_mask,
        area,
    )


def extract_segmentation_flag_scores(
    image_paths: list[Path],
    device: str,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    min_area_ratio: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    processor, model, label_to_id = load_segformer_ade20k_runtime(device=device)
    runtime_device = device
    class_names = {int(class_id): str(label) for label, class_id in label_to_id.items()}
    prompt_class_lookup = build_segformer_prompt_class_lookup(prompt_texts, class_names)

    combined_scores: list[np.ndarray] = []
    combined_regions: list[list[dict | None]] = []
    valid_paths: list[Path] = []
    min_area_ratio = max(0.0, float(min_area_ratio))

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            cached = feature_cache.get(image_path) if feature_cache is not None else None
            image = cached.image if cached is not None else load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during segmentation flag extraction: %s", image_path)
                continue

            image_scores = np.zeros(len(prompt_texts), dtype=np.float32)
            image_regions: list[dict | None] = [None for _ in prompt_texts]

            if prompt_class_lookup:
                try:
                    class_probabilities, probability_size = run_segformer_ade20k_inference(
                        processor=processor,
                        model=model,
                        image=image,
                        device=runtime_device,
                    )
                except torch.OutOfMemoryError:
                    if runtime_device != "cpu":
                        logger.warning(
                            "SegFormer ran out of GPU memory on %s. Retrying segmentation on CPU for the remaining images.",
                            image_path.name,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        processor, model, label_to_id = load_segformer_ade20k_runtime(device="cpu")
                        runtime_device = "cpu"
                        class_probabilities, probability_size = run_segformer_ade20k_inference(
                            processor=processor,
                            model=model,
                            image=image,
                            device=runtime_device,
                        )
                    else:
                        raise
                predicted_classes = class_probabilities.argmax(axis=0)
                probability_width, probability_height = probability_size
                total_pixels = max(1, probability_width * probability_height)

                for class_id, prompt_indices in prompt_class_lookup.items():
                    region, component_mask, component_area = largest_component_region(predicted_classes == class_id)
                    if region is None or component_mask is None:
                        continue

                    area_ratio = float(component_area / total_pixels)
                    if area_ratio < min_area_ratio:
                        continue

                    confidence = float(np.clip(class_probabilities[class_id][component_mask].mean(), 0.0, 1.0))
                    scaled_region = scale_region_to_image(
                        region=region,
                        source_width=probability_width,
                        source_height=probability_height,
                        target_width=image.width,
                        target_height=image.height,
                    )
                    scaled_boundary = scale_boundary_to_image(
                        boundary=mask_to_boundary_points(component_mask),
                        source_width=probability_width,
                        source_height=probability_height,
                        target_width=image.width,
                        target_height=image.height,
                    )
                    region_payload = build_region_payload(
                        int(scaled_region["left"]),
                        int(scaled_region["top"]),
                        int(scaled_region["right"]),
                        int(scaled_region["bottom"]),
                        boundary=scaled_boundary,
                        extras={
                            "coverage_percent": round(area_ratio * 100.0, 2),
                            "source": "segformer_ade20k",
                        },
                    )
                    for prompt_index in prompt_indices:
                        image_scores[prompt_index] = confidence
                        image_regions[prompt_index] = dict(region_payload)

            combined_scores.append(image_scores)
            combined_regions.append(image_regions)
            valid_paths.append(image_path)

            if index % 4 == 0 or index == len(image_paths):
                logger.info("Scored segmentation item flags for %s/%s images", index, len(image_paths))

    if not combined_scores:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.stack(combined_scores).astype(np.float32, copy=False), combined_regions, valid_paths


def extract_deeplab_flag_scores(
    image_paths: list[Path],
    device: str,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    model_name: str,
    min_area_ratio: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    preprocess, model, class_names = load_deeplabv3_runtime(model_name=model_name, device=device)
    runtime_device = device
    prompt_class_lookup = build_deeplab_prompt_class_lookup(prompt_texts, class_names)
    min_area_ratio = max(0.0, float(min_area_ratio))

    combined_scores: list[np.ndarray] = []
    combined_regions: list[list[dict | None]] = []
    valid_paths: list[Path] = []

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            cached = feature_cache.get(image_path) if feature_cache is not None else None
            image = cached.image if cached is not None else load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during DeepLabV3 flag extraction: %s", image_path)
                continue

            image_scores = np.zeros(len(prompt_texts), dtype=np.float32)
            image_regions: list[dict | None] = [None for _ in prompt_texts]

            if prompt_class_lookup:
                try:
                    class_probabilities, probability_size = run_deeplabv3_inference(
                        preprocess=preprocess,
                        model=model,
                        image=image,
                        device=runtime_device,
                    )
                except torch.OutOfMemoryError:
                    if runtime_device != "cpu":
                        logger.warning(
                            "DeepLabV3 ran out of GPU memory on %s. Retrying DeepLabV3 on CPU for the remaining images.",
                            image_path.name,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        preprocess, model, class_names = load_deeplabv3_runtime(model_name=model_name, device="cpu")
                        runtime_device = "cpu"
                        class_probabilities, probability_size = run_deeplabv3_inference(
                            preprocess=preprocess,
                            model=model,
                            image=image,
                            device=runtime_device,
                        )
                    else:
                        raise

                predicted_classes = class_probabilities.argmax(axis=0)
                probability_width, probability_height = probability_size
                total_pixels = max(1, probability_width * probability_height)

                for class_id, prompt_indices in prompt_class_lookup.items():
                    region, component_mask, component_area = largest_component_region(predicted_classes == class_id)
                    if region is None or component_mask is None:
                        continue

                    area_ratio = float(component_area / total_pixels)
                    if area_ratio < min_area_ratio:
                        continue

                    confidence = float(np.clip(class_probabilities[class_id][component_mask].mean(), 0.0, 1.0))
                    scaled_region = scale_region_to_image(
                        region=region,
                        source_width=probability_width,
                        source_height=probability_height,
                        target_width=image.width,
                        target_height=image.height,
                    )
                    scaled_boundary = scale_boundary_to_image(
                        boundary=mask_to_boundary_points(component_mask),
                        source_width=probability_width,
                        source_height=probability_height,
                        target_width=image.width,
                        target_height=image.height,
                    )
                    region_payload = build_region_payload(
                        int(scaled_region["left"]),
                        int(scaled_region["top"]),
                        int(scaled_region["right"]),
                        int(scaled_region["bottom"]),
                        boundary=scaled_boundary,
                        extras={
                            "coverage_percent": round(area_ratio * 100.0, 2),
                            "source": "deeplabv3",
                            "deeplab_model": model_name,
                            "deeplab_class": class_names.get(class_id, str(class_id)),
                        },
                    )
                    for prompt_index in prompt_indices:
                        image_scores[prompt_index] = confidence
                        image_regions[prompt_index] = dict(region_payload)

            combined_scores.append(image_scores)
            combined_regions.append(image_regions)
            valid_paths.append(image_path)

            if index % 4 == 0 or index == len(image_paths):
                logger.info("Scored DeepLabV3 item flags for %s/%s images", index, len(image_paths))

    if not combined_scores:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.stack(combined_scores).astype(np.float32, copy=False), combined_regions, valid_paths


def extract_hybrid_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    yolo_model_path: str,
    yolo_confidence_threshold: float,
    yolo_iou_threshold: float,
    yolo_image_size: int,
    yolo_max_detections: int,
    yolo_retina_masks: bool,
    open_vocab_model_id: str,
    open_vocab_score_threshold: float,
    segmentation_min_area_ratio: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    backend_outputs: list[tuple[np.ndarray, list[list[dict | None]] | None, list[Path]]] = []
    fallback_score_floor = max(
        HYBRID_FALLBACK_SCORE_FLOOR,
        min(float(yolo_confidence_threshold), float(open_vocab_score_threshold)),
    )

    yolo_model = load_yolo_runtime(model_path=yolo_model_path)
    yolo_class_names = resolve_named_class_lookup(
        getattr(yolo_model, "names", None) or getattr(getattr(yolo_model, "model", None), "names", None)
    )
    yolo_prompt_lookup = build_yolo_prompt_class_lookup(prompt_texts, yolo_class_names)
    yolo_prompt_indices = prompt_indices_from_class_lookup(yolo_prompt_lookup)
    yolo_scores = np.zeros((len(image_paths), len(prompt_texts)), dtype=np.float32)
    yolo_regions: list[list[dict | None]] = [[None for _ in prompt_texts] for _ in image_paths]
    yolo_paths = list(image_paths)
    if yolo_prompt_indices:
        yolo_scores, yolo_regions, yolo_paths = extract_yolo_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence_threshold,
            iou_threshold=yolo_iou_threshold,
            image_size=yolo_image_size,
            max_detections=yolo_max_detections,
            retina_masks=yolo_retina_masks,
        )
        backend_outputs.append((yolo_scores, yolo_regions, yolo_paths))

    fallback_object_paths = select_low_coverage_image_paths(
        image_paths=yolo_paths,
        scores=yolo_scores,
        regions=yolo_regions,
        prompt_texts=prompt_texts,
        min_score=fallback_score_floor,
        min_localized_non_scene=HYBRID_FALLBACK_MIN_LOCALIZED_OBJECT_FLAGS,
    )
    open_vocab_prompt_indices = [
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if (
            prompt_text_to_label(prompt_text).lower() not in HYBRID_CLIP_LABELS
            and prompt_text_to_label(prompt_text).lower() not in SCENE_FLAG_LABELS
        )
    ]
    if open_vocab_prompt_indices and fallback_object_paths:
        open_vocab_prompt_texts = [prompt_texts[prompt_index] for prompt_index in open_vocab_prompt_indices]
        open_vocab_scores_subset, open_vocab_regions_subset, open_vocab_paths = extract_open_vocab_flag_scores(
            image_paths=fallback_object_paths,
            device=device,
            prompt_texts=open_vocab_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_id=open_vocab_model_id,
            score_threshold=open_vocab_score_threshold,
        )
        open_vocab_scores, open_vocab_regions = expand_prompt_backend_output(
            scores_subset=open_vocab_scores_subset,
            regions_subset=open_vocab_regions_subset,
            subset_prompt_indices=open_vocab_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=open_vocab_paths,
        )
        backend_outputs.append((open_vocab_scores, open_vocab_regions, open_vocab_paths))

    _, _, segformer_label_to_id = load_segformer_ade20k_runtime(device=device)
    segformer_class_names = {int(class_id): str(label) for label, class_id in segformer_label_to_id.items()}
    segformer_prompt_lookup = build_segformer_prompt_class_lookup(prompt_texts, segformer_class_names)
    segformer_prompt_indices = prompt_indices_from_class_lookup(segformer_prompt_lookup)
    if segformer_prompt_indices:
        seg_scores, seg_regions, seg_paths = extract_segmentation_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            min_area_ratio=segmentation_min_area_ratio,
        )
        backend_outputs.append((seg_scores, seg_regions, seg_paths))

    merged_scores, merged_regions = merge_flag_backend_outputs(
        image_paths=image_paths,
        prompt_count=len(prompt_texts),
        backend_outputs=backend_outputs,
    )
    clip_fallback_paths = select_low_coverage_image_paths(
        image_paths=image_paths,
        scores=merged_scores,
        regions=merged_regions,
        prompt_texts=prompt_texts,
        min_score=HYBRID_FALLBACK_SCORE_FLOOR,
        min_localized_non_scene=1,
    )
    clip_prompt_indices = {
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if (
            prompt_index not in open_vocab_prompt_indices
            and prompt_index not in segformer_prompt_indices
        )
        or prompt_text_to_label(prompt_text).lower() in HYBRID_CLIP_LABELS
    }
    if clip_prompt_indices and clip_fallback_paths:
        clip_prompt_indices = set(sorted(clip_prompt_indices))
        expanded_clip_prompts: list[str] = []
        expanded_prompt_targets: list[int | None] = []
        requested_scene_labels = {
            prompt_text_to_label(prompt_texts[prompt_index]).lower(): prompt_index for prompt_index in sorted(clip_prompt_indices)
        }
        for prompt_index in sorted(clip_prompt_indices):
            prompt_text = prompt_texts[prompt_index]
            normalized_label = prompt_text_to_label(prompt_text).lower()
            if normalized_label == "outdoor":
                expanded_clip_prompts.append("an outdoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "indoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an indoor scene")
                    expanded_prompt_targets.append(None)
                continue
            if normalized_label == "indoor":
                expanded_clip_prompts.append("an indoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "outdoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an outdoor scene")
                    expanded_prompt_targets.append(None)
                continue
            expanded_clip_prompts.append(prompt_text)
            expanded_prompt_targets.append(prompt_index)

        clip_scores_subset, clip_regions_subset, clip_paths = extract_clip_flag_scores(
            image_paths=image_paths,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=expanded_clip_prompts,
            feature_cache=feature_cache,
            logger=logger,
        )
        full_clip_scores = np.zeros((len(clip_paths), len(prompt_texts)), dtype=np.float32)
        full_clip_regions: list[list[dict | None]] = [[None for _ in prompt_texts] for _ in clip_paths]
        for subset_index, prompt_index in enumerate(expanded_prompt_targets):
            if prompt_index is None:
                continue
            if clip_scores_subset.size:
                full_clip_scores[:, prompt_index] = clip_scores_subset[:, subset_index]
            if clip_regions_subset:
                for image_index in range(min(len(clip_paths), len(clip_regions_subset))):
                    region = clip_regions_subset[image_index][subset_index]
                    if region is not None:
                        full_clip_regions[image_index][prompt_index] = dict(region)
        backend_outputs.append((full_clip_scores, full_clip_regions, clip_paths))
        resolved_paths = clip_paths

    merged_scores, merged_regions = merge_flag_backend_outputs(
        image_paths=image_paths,
        prompt_count=len(prompt_texts),
        backend_outputs=backend_outputs,
    )
    return merged_scores, merged_regions, resolved_paths


def extract_sam_deeplab_yolo_clip_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    yolo_model_path: str,
    yolo_confidence_threshold: float,
    yolo_iou_threshold: float,
    yolo_image_size: int,
    yolo_max_detections: int,
    yolo_retina_masks: bool,
    deeplab_model_name: str,
    deeplab_min_area_ratio: float,
    sam_model_path: str,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    backend_outputs: list[tuple[np.ndarray, list[list[dict | None]] | None, list[Path]]] = []

    yolo_scores, yolo_regions, yolo_paths = extract_yolo_flag_scores(
        image_paths=image_paths,
        device=device,
        prompt_texts=prompt_texts,
        feature_cache=feature_cache,
        logger=logger,
        model_path=yolo_model_path,
        confidence_threshold=yolo_confidence_threshold,
        iou_threshold=yolo_iou_threshold,
        image_size=yolo_image_size,
        max_detections=yolo_max_detections,
        retina_masks=yolo_retina_masks,
    )
    backend_outputs.append((yolo_scores, yolo_regions, yolo_paths))

    deeplab_scores, deeplab_regions, deeplab_paths = extract_deeplab_flag_scores(
        image_paths=image_paths,
        device=device,
        prompt_texts=prompt_texts,
        feature_cache=feature_cache,
        logger=logger,
        model_name=deeplab_model_name,
        min_area_ratio=deeplab_min_area_ratio,
    )
    backend_outputs.append((deeplab_scores, deeplab_regions, deeplab_paths))

    yolo_model = load_yolo_runtime(model_path=yolo_model_path)
    yolo_class_names = resolve_named_class_lookup(
        getattr(yolo_model, "names", None) or getattr(getattr(yolo_model, "model", None), "names", None)
    )
    yolo_prompt_indices = prompt_indices_from_class_lookup(build_yolo_prompt_class_lookup(prompt_texts, yolo_class_names))

    deeplab_class_names = resolve_deeplabv3_class_names(deeplab_model_name)
    deeplab_prompt_indices = prompt_indices_from_class_lookup(
        build_deeplab_prompt_class_lookup(prompt_texts, deeplab_class_names)
    )

    merged_scores, merged_regions = merge_flag_backend_outputs(
        image_paths=image_paths,
        prompt_count=len(prompt_texts),
        backend_outputs=backend_outputs,
    )
    clip_fallback_paths = select_low_coverage_image_paths(
        image_paths=image_paths,
        scores=merged_scores,
        regions=merged_regions,
        prompt_texts=prompt_texts,
        min_score=HYBRID_FALLBACK_SCORE_FLOOR,
        min_localized_non_scene=1,
    )
    clip_prompt_indices = {
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if (
            prompt_index not in yolo_prompt_indices
            and prompt_index not in deeplab_prompt_indices
        )
        or prompt_text_to_label(prompt_text).lower() in HYBRID_CLIP_LABELS
        or prompt_text_to_label(prompt_text).lower() in SCENE_FLAG_LABELS
    }
    if clip_prompt_indices and clip_fallback_paths:
        clip_prompt_indices = sorted(clip_prompt_indices)
        clip_prompt_texts = [prompt_texts[prompt_index] for prompt_index in clip_prompt_indices]
        clip_scores_subset, clip_regions_subset, clip_paths = extract_clip_flag_scores(
            image_paths=clip_fallback_paths,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=clip_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
        )
        clip_scores, clip_regions = expand_prompt_backend_output(
            scores_subset=clip_scores_subset,
            regions_subset=clip_regions_subset,
            subset_prompt_indices=clip_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=clip_paths,
        )
        backend_outputs.append((clip_scores, clip_regions, clip_paths))
        merged_scores, merged_regions = merge_flag_backend_outputs(
            image_paths=image_paths,
            prompt_count=len(prompt_texts),
            backend_outputs=backend_outputs,
        )

    merged_regions = refine_regions_with_sam(
        image_paths=image_paths,
        device=device,
        prompt_texts=prompt_texts,
        prompt_scores=merged_scores,
        prompt_regions=merged_regions,
        feature_cache=feature_cache,
        logger=logger,
        model_path=sam_model_path,
        min_score=HYBRID_FALLBACK_SCORE_FLOOR,
    )
    return merged_scores, merged_regions, list(image_paths)


def expand_prompt_backend_output(
    scores_subset: np.ndarray,
    regions_subset: list[list[dict | None]],
    subset_prompt_indices: list[int],
    prompt_count: int,
    resolved_paths: list[Path],
) -> tuple[np.ndarray, list[list[dict | None]]]:
    expanded_scores = np.zeros((len(resolved_paths), prompt_count), dtype=np.float32)
    expanded_regions: list[list[dict | None]] = [[None for _ in range(prompt_count)] for _ in resolved_paths]
    if scores_subset.size == 0:
        return expanded_scores, expanded_regions

    for subset_index, prompt_index in enumerate(subset_prompt_indices):
        if prompt_index < 0 or prompt_index >= prompt_count:
            continue
        expanded_scores[:, prompt_index] = scores_subset[:, subset_index]
        for image_index in range(min(len(resolved_paths), len(regions_subset))):
            if subset_index >= len(regions_subset[image_index]):
                continue
            region = regions_subset[image_index][subset_index]
            if region is not None:
                expanded_regions[image_index][prompt_index] = dict(region)
    return expanded_scores, expanded_regions


def select_prompt_indices_for_labels(prompt_texts: list[str], allowed_labels: set[str]) -> list[int]:
    normalized_allowed_labels = {normalize_detector_label(label) for label in allowed_labels if normalize_detector_label(label)}
    return [
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if normalize_detector_label(prompt_text_to_label(prompt_text)) in normalized_allowed_labels
    ]


def generate_scene_region_candidates(
    image: Image.Image,
    prompt_texts: list[str],
) -> list[dict[str, object]]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return []

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return []

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    image_area = max(1, int(width * height))
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    available_labels = {
        normalize_detector_label(prompt_text_to_label(prompt_text))
        for prompt_text in prompt_texts
        if normalize_detector_label(prompt_text_to_label(prompt_text))
    }
    if not available_labels:
        return []

    candidates: list[dict[str, object]] = []
    seen_candidates: set[tuple[int, int, int, int, tuple[str, ...]]] = set()

    def append_candidate(
        *,
        left: int,
        top: int,
        right: int,
        bottom: int,
        boundary: list[dict[str, int]] | None,
        labels: set[str],
        heuristic_scores: dict[str, float],
        source: str,
    ) -> None:
        normalized_labels = sorted(
            {
                normalize_detector_label(label)
                for label in labels
                if normalize_detector_label(label) in available_labels
            }
        )
        if not normalized_labels:
            return
        left = max(0, min(int(left), max(0, width - 1)))
        top = max(0, min(int(top), max(0, height - 1)))
        right = max(left + 1, min(int(right), width))
        bottom = max(top + 1, min(int(bottom), height))
        candidate_key = (left, top, right, bottom, tuple(normalized_labels))
        if candidate_key in seen_candidates:
            return
        seen_candidates.add(candidate_key)
        filtered_scores = {
            label: float(np.clip(heuristic_scores.get(label, 0.0), 0.0, 1.0))
            for label in normalized_labels
            if heuristic_scores.get(label, 0.0) > 0.0
        }
        candidates.append(
            {
                "region": build_region_payload(
                    left,
                    top,
                    right,
                    bottom,
                    boundary=boundary or box_to_boundary_points(left, top, right, bottom),
                    extras={
                        "source": "scene_heuristic",
                        "scene_candidate_source": source,
                    },
                ),
                "prompt_labels": normalized_labels,
                "heuristic_scores": filtered_scores,
            }
        )

    def append_mask_candidate(
        *,
        mask: np.ndarray,
        labels: set[str],
        heuristic_scores: dict[str, float],
        source: str,
        min_area_ratio: float = 0.012,
    ) -> dict[str, int] | None:
        mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
        if mask_uint8.ndim != 2 or not mask_uint8.any():
            return None
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        region, component_mask, component_area = largest_component_region(mask_uint8.astype(bool))
        if region is None or component_mask is None:
            return None
        if component_area < int(image_area * max(0.0, float(min_area_ratio))):
            return None
        boundary = mask_to_boundary_points(component_mask)
        append_candidate(
            left=int(region["left"]),
            top=int(region["top"]),
            right=int(region["right"]),
            bottom=int(region["bottom"]),
            boundary=boundary,
            labels=labels,
            heuristic_scores=heuristic_scores,
            source=source,
        )
        return {
            "left": int(region["left"]),
            "top": int(region["top"]),
            "right": int(region["right"]),
            "bottom": int(region["bottom"]),
        }

    def append_box_candidate(
        *,
        box: tuple[int, int, int, int],
        labels: set[str],
        heuristic_scores: dict[str, float],
        source: str,
    ) -> None:
        left, top, right, bottom = box
        append_candidate(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            boundary=box_to_boundary_points(left, top, right, bottom),
            labels=labels,
            heuristic_scores=heuristic_scores,
            source=source,
        )

    bright_mask = value > 172.0
    blue_sky_mask = (blue > green + 8) & (blue > red + 8) & (blue > 120)
    opening_mask = ((bright_mask & (saturation < 110.0)) | blue_sky_mask) & (y_coords < int(height * 0.94))
    opening_region = append_mask_candidate(
        mask=opening_mask,
        labels={
            "window",
            "sliding glass door",
            "outdoor",
            "sky",
            "clouds",
            "tree",
            "trees",
            "grass",
            "lawn",
            "building facade",
            "balcony railing",
            "patio",
            "driveway",
            "fence",
            "swimming pool",
        },
        heuristic_scores={
            "window": 0.22,
            "sliding glass door": 0.20,
            "outdoor": 0.20,
            "sky": 0.18,
            "clouds": 0.16,
            "tree": 0.14,
            "trees": 0.14,
            "grass": 0.12,
            "lawn": 0.12,
            "building facade": 0.10,
            "balcony railing": 0.10,
            "patio": 0.09,
            "driveway": 0.08,
            "fence": 0.08,
        },
        source="opening_mask",
        min_area_ratio=0.010,
    )

    sky_mask = (
        blue_sky_mask
        | (
            (hsv[:, :, 0] > 85)
            & (hsv[:, :, 0] < 135)
            & (saturation > 20.0)
            & (value > 110.0)
        )
    ) & (y_coords < int(height * 0.72))
    append_mask_candidate(
        mask=sky_mask,
        labels={"sky", "clouds", "outdoor"},
        heuristic_scores={"sky": 0.24, "clouds": 0.18, "outdoor": 0.18},
        source="sky_mask",
        min_area_ratio=0.010,
    )

    white_surface_mask = (saturation < 30.0) & (value > 160.0) & ~opening_mask
    append_mask_candidate(
        mask=white_surface_mask & (x_coords < int(width * 0.48)) & (y_coords > int(height * 0.10)) & (y_coords < int(height * 0.92)),
        labels={"wall", "white wall"},
        heuristic_scores={"wall": 0.18, "white wall": 0.24},
        source="left_wall_mask",
        min_area_ratio=0.025,
    )
    append_mask_candidate(
        mask=white_surface_mask & (x_coords > int(width * 0.52)) & (y_coords > int(height * 0.10)) & (y_coords < int(height * 0.92)),
        labels={"wall", "white wall"},
        heuristic_scores={"wall": 0.18, "white wall": 0.24},
        source="right_wall_mask",
        min_area_ratio=0.025,
    )
    append_mask_candidate(
        mask=white_surface_mask & (y_coords < int(height * 0.28)),
        labels={"wall", "white wall", "ceiling lights"},
        heuristic_scores={"wall": 0.14, "white wall": 0.20, "ceiling lights": 0.14},
        source="ceiling_mask",
        min_area_ratio=0.015,
    )

    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    floor_texture = cv2.GaussianBlur(laplacian, (5, 5), 0)
    floor_texture_threshold = max(8.0, float(np.percentile(floor_texture, 65)))
    floor_mask = (
        (y_coords > int(height * 0.58))
        & ~opening_mask
        & (value > 70.0)
        & ((saturation < 95.0) | (floor_texture >= floor_texture_threshold))
    )
    append_mask_candidate(
        mask=floor_mask,
        labels={"floor", "floor tiles"},
        heuristic_scores={"floor": 0.19, "floor tiles": 0.23},
        source="floor_mask",
        min_area_ratio=0.045,
    )

    append_box_candidate(
        box=(0, 0, width, max(1, int(height * 0.30))),
        labels={"wall", "white wall", "ceiling lights", "sky", "clouds"},
        heuristic_scores={"wall": 0.10, "white wall": 0.12, "ceiling lights": 0.10, "sky": 0.08},
        source="top_band",
    )
    append_box_candidate(
        box=(0, int(height * 0.58), width, height),
        labels={"floor", "floor tiles"},
        heuristic_scores={"floor": 0.12, "floor tiles": 0.14},
        source="bottom_band",
    )
    append_box_candidate(
        box=(0, int(height * 0.12), max(1, int(width * 0.44)), int(height * 0.92)),
        labels={"wall", "white wall"},
        heuristic_scores={"wall": 0.11, "white wall": 0.14},
        source="left_band",
    )
    append_box_candidate(
        box=(int(width * 0.56), int(height * 0.12), width, int(height * 0.92)),
        labels={"wall", "white wall"},
        heuristic_scores={"wall": 0.11, "white wall": 0.14},
        source="right_band",
    )

    if opening_region is not None:
        opening_left = int(opening_region["left"])
        opening_top = int(opening_region["top"])
        opening_right = int(opening_region["right"])
        opening_bottom = int(opening_region["bottom"])
        opening_width = max(1, opening_right - opening_left)
        opening_height = max(1, opening_bottom - opening_top)
        expand_x = max(12, int(width * 0.04))
        expand_y = max(12, int(height * 0.04))
        expanded_box = (
            max(0, opening_left - expand_x),
            max(0, opening_top - expand_y),
            min(width, opening_right + expand_x),
            min(height, opening_bottom + expand_y),
        )
        append_box_candidate(
            box=(opening_left, opening_top, opening_right, opening_bottom),
            labels={
                "window",
                "sliding glass door",
                "outdoor",
                "sky",
                "clouds",
                "tree",
                "trees",
                "grass",
                "lawn",
                "building facade",
                "balcony railing",
                "patio",
                "driveway",
                "fence",
                "swimming pool",
            },
            heuristic_scores={
                "window": 0.24,
                "sliding glass door": 0.23,
                "outdoor": 0.20,
                "sky": 0.18,
                "clouds": 0.16,
                "tree": 0.16,
                "trees": 0.16,
                "grass": 0.15,
                "lawn": 0.15,
                "balcony railing": 0.15,
                "patio": 0.14,
                "driveway": 0.12,
                "fence": 0.12,
            },
            source="opening_box",
        )
        append_box_candidate(
            box=expanded_box,
            labels={
                "window",
                "sliding glass door",
                "outdoor",
                "sky",
                "clouds",
                "tree",
                "trees",
                "grass",
                "lawn",
                "building facade",
                "balcony railing",
                "patio",
                "driveway",
                "fence",
                "swimming pool",
            },
            heuristic_scores={
                "window": 0.18,
                "sliding glass door": 0.18,
                "outdoor": 0.18,
                "sky": 0.16,
                "tree": 0.14,
                "trees": 0.14,
                "grass": 0.12,
                "lawn": 0.12,
            },
            source="opening_context",
        )

        upper_opening_box = (
            opening_left,
            opening_top,
            opening_right,
            max(opening_top + 1, opening_top + int(opening_height * 0.45)),
        )
        append_box_candidate(
            box=upper_opening_box,
            labels={"window", "sliding glass door", "sky", "clouds", "outdoor", "building facade"},
            heuristic_scores={
                "window": 0.20,
                "sliding glass door": 0.18,
                "sky": 0.24,
                "clouds": 0.20,
                "outdoor": 0.18,
                "building facade": 0.12,
            },
            source="opening_upper",
        )

        lower_opening_box = (
            opening_left,
            min(height - 1, opening_top + int(opening_height * 0.42)),
            opening_right,
            opening_bottom,
        )
        append_box_candidate(
            box=lower_opening_box,
            labels={
                "window",
                "sliding glass door",
                "outdoor",
                "tree",
                "trees",
                "grass",
                "lawn",
                "balcony railing",
                "patio",
                "driveway",
                "fence",
                "swimming pool",
            },
            heuristic_scores={
                "window": 0.16,
                "sliding glass door": 0.18,
                "outdoor": 0.18,
                "tree": 0.20,
                "trees": 0.20,
                "grass": 0.20,
                "lawn": 0.20,
                "balcony railing": 0.18,
                "patio": 0.16,
                "driveway": 0.15,
                "fence": 0.14,
                "swimming pool": 0.12,
            },
            source="opening_lower",
        )

        vertical_opening_box = (
            opening_left + int(opening_width * 0.18),
            opening_top,
            max(opening_left + int(opening_width * 0.82), opening_left + 1),
            opening_bottom,
        )
        append_box_candidate(
            box=vertical_opening_box,
            labels={"window", "sliding glass door", "outdoor", "sky", "tree", "trees", "grass", "lawn", "balcony railing"},
            heuristic_scores={
                "window": 0.22,
                "sliding glass door": 0.22,
                "outdoor": 0.18,
                "sky": 0.18,
                "tree": 0.17,
                "trees": 0.17,
                "grass": 0.17,
                "lawn": 0.17,
                "balcony railing": 0.16,
            },
            source="opening_vertical",
        )

    if not candidates:
        append_box_candidate(
            box=(0, 0, width, height),
            labels=available_labels,
            heuristic_scores={},
            source="full_image",
        )

    return candidates


def extract_scene_clip_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    *,
    min_score: float = DEFAULT_SCENE_CLIP_MIN_SCORE,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths or not prompt_texts:
        return np.empty((0, 0), dtype=np.float32), [], []

    clip, model, preprocess = load_clip_runtime(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        logger=logger,
    )
    min_score = float(np.clip(min_score, 0.0, 1.0))
    normalized_prompt_labels = [
        normalize_detector_label(prompt_text_to_label(prompt_text))
        for prompt_text in prompt_texts
    ]

    with torch.no_grad():
        prompt_tokens = clip.tokenize(prompt_texts).to(device)
        prompt_features = model.encode_text(prompt_tokens).float()
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

    combined_scores: list[np.ndarray] = []
    combined_regions: list[list[dict | None]] = []
    valid_paths: list[Path] = []

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            cached = feature_cache.get(image_path) if feature_cache is not None else None
            image = cached.image if cached is not None else load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during scene CLIP labeling: %s", image_path)
                continue

            candidates = generate_scene_region_candidates(image, prompt_texts)
            if not candidates:
                continue

            crops: list[Image.Image] = []
            for candidate in candidates:
                region = candidate["region"]
                left = int(region["left"])
                top = int(region["top"])
                right = int(region["right"])
                bottom = int(region["bottom"])
                crops.append(image.crop((left, top, right, bottom)))

            crop_tensor = torch.stack([preprocess(crop) for crop in crops]).to(device)
            crop_features = model.encode_image(crop_tensor).float()
            crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
            similarity = (crop_features @ prompt_features.T).cpu().numpy().astype(np.float32, copy=False)

            image_scores = np.zeros(len(prompt_texts), dtype=np.float32)
            image_regions: list[dict | None] = [None for _ in prompt_texts]
            for candidate_index, candidate in enumerate(candidates):
                candidate_labels = set(candidate["prompt_labels"])
                heuristic_scores = {
                    normalize_detector_label(label): float(np.clip(score, 0.0, 1.0))
                    for label, score in dict(candidate.get("heuristic_scores", {})).items()
                }
                region = dict(candidate["region"])
                candidate_source = str(region.get("scene_candidate_source", "heuristic"))
                for prompt_index, normalized_label in enumerate(normalized_prompt_labels):
                    if not normalized_label or normalized_label not in candidate_labels:
                        continue
                    clip_score = float(np.clip(similarity[candidate_index, prompt_index], 0.0, 1.0))
                    heuristic_score = float(heuristic_scores.get(normalized_label, 0.0))
                    combined_score = max(clip_score, heuristic_score)
                    if combined_score < min_score:
                        continue
                    if combined_score <= float(image_scores[prompt_index]):
                        continue
                    region_payload = dict(region)
                    region_payload["source"] = "scene_clip" if clip_score >= heuristic_score else "scene_heuristic"
                    region_payload["scene_candidate_source"] = candidate_source
                    region_payload["clip_similarity"] = similarity_to_percent(clip_score)
                    region_payload["prompt_region_score_percent"] = similarity_to_percent(combined_score)
                    image_scores[prompt_index] = combined_score
                    image_regions[prompt_index] = region_payload

            combined_scores.append(image_scores)
            combined_regions.append(image_regions)
            valid_paths.append(image_path)

            if index % 8 == 0 or index == len(image_paths):
                logger.info("Scored heuristic scene CLIP labels for %s/%s images", index, len(image_paths))

    if not combined_scores:
        return np.empty((0, 0), dtype=np.float32), [], []

    return np.stack(combined_scores).astype(np.float32, copy=False), combined_regions, valid_paths


def extract_yolo_scene_clip_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    yolo_model_path: str,
    yolo_confidence_threshold: float,
    yolo_iou_threshold: float,
    yolo_image_size: int,
    yolo_max_detections: int,
    yolo_retina_masks: bool,
    scene_clip_min_score: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    backend_outputs: list[tuple[np.ndarray, list[list[dict | None]] | None, list[Path]]] = []

    yolo_prompt_indices = select_prompt_indices_for_labels(prompt_texts, YOLO_SCENE_CLIP_OBJECT_LABELS)
    if yolo_prompt_indices:
        yolo_prompt_texts = [prompt_texts[prompt_index] for prompt_index in yolo_prompt_indices]
        yolo_scores_subset, yolo_regions_subset, yolo_paths = extract_yolo_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=yolo_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence_threshold,
            iou_threshold=yolo_iou_threshold,
            image_size=yolo_image_size,
            max_detections=yolo_max_detections,
            retina_masks=yolo_retina_masks,
        )
        yolo_scores, yolo_regions = expand_prompt_backend_output(
            scores_subset=yolo_scores_subset,
            regions_subset=yolo_regions_subset,
            subset_prompt_indices=yolo_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=yolo_paths,
        )
        backend_outputs.append((yolo_scores, yolo_regions, yolo_paths))

    scene_prompt_indices = select_prompt_indices_for_labels(prompt_texts, YOLO_SCENE_CLIP_SCENE_LABELS)
    if scene_prompt_indices:
        scene_prompt_texts = [prompt_texts[prompt_index] for prompt_index in scene_prompt_indices]
        scene_scores_subset, scene_regions_subset, scene_paths = extract_scene_clip_flag_scores(
            image_paths=image_paths,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=scene_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            min_score=scene_clip_min_score,
        )
        scene_scores, scene_regions = expand_prompt_backend_output(
            scores_subset=scene_scores_subset,
            regions_subset=scene_regions_subset,
            subset_prompt_indices=scene_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=scene_paths,
        )
        backend_outputs.append((scene_scores, scene_regions, scene_paths))

    merged_scores, merged_regions = merge_flag_backend_outputs(
        image_paths=image_paths,
        prompt_count=len(prompt_texts),
        backend_outputs=backend_outputs,
    )
    return merged_scores, merged_regions, list(image_paths)


def region_area(region: dict[str, object] | None) -> int:
    if not isinstance(region, dict):
        return 0
    try:
        left = int(region.get("left", 0))
        top = int(region.get("top", 0))
        right = int(region.get("right", 0))
        bottom = int(region.get("bottom", 0))
    except (TypeError, ValueError):
        return 0
    return max(0, right - left) * max(0, bottom - top)


def region_iou(first_region: dict[str, object] | None, second_region: dict[str, object] | None) -> float:
    if not isinstance(first_region, dict) or not isinstance(second_region, dict):
        return 0.0
    try:
        first_left = int(first_region.get("left", 0))
        first_top = int(first_region.get("top", 0))
        first_right = int(first_region.get("right", 0))
        first_bottom = int(first_region.get("bottom", 0))
        second_left = int(second_region.get("left", 0))
        second_top = int(second_region.get("top", 0))
        second_right = int(second_region.get("right", 0))
        second_bottom = int(second_region.get("bottom", 0))
    except (TypeError, ValueError):
        return 0.0

    intersection_left = max(first_left, second_left)
    intersection_top = max(first_top, second_top)
    intersection_right = min(first_right, second_right)
    intersection_bottom = min(first_bottom, second_bottom)
    intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
    if intersection_area <= 0:
        return 0.0

    first_area = max(0, first_right - first_left) * max(0, first_bottom - first_top)
    second_area = max(0, second_right - second_left) * max(0, second_bottom - second_top)
    denominator = first_area + second_area - intersection_area
    if denominator <= 0:
        return 0.0
    return float(intersection_area / denominator)


def scene_candidate_source_priority(source: str) -> int:
    normalized_source = normalize_detector_label(source)
    priority = {
        "sky mask": 0,
        "floor mask": 0,
        "left wall mask": 0,
        "right wall mask": 0,
        "opening mask": 1,
        "opening upper": 1,
        "opening lower": 1,
        "opening vertical": 1,
        "opening box": 2,
        "top band": 3,
        "bottom band": 3,
        "left band": 3,
        "right band": 3,
        "opening context": 4,
        "full image": 5,
    }
    return priority.get(normalized_source, 6)


def scene_duplicate_limit(label: str) -> int:
    normalized_label = normalize_detector_label(label)
    if normalized_label in {"tree", "trees"}:
        return 3
    if normalized_label in {"white wall", "wall", "window", "floor tiles", "floor"}:
        return 2
    return 1


def scene_label_score_floor(label: str, base_min_score: float) -> float:
    normalized_label = normalize_detector_label(label)
    return max(float(base_min_score), float(SCENE_LABEL_SCORE_FLOORS.get(normalized_label, base_min_score)))


def extract_scene_clip_flag_items(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    *,
    min_score: float = DEFAULT_SCENE_CLIP_MIN_SCORE,
) -> tuple[list[list[dict]], list[Path]]:
    if not image_paths or not prompt_texts:
        return [], []

    clip, model, preprocess = load_clip_runtime(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        logger=logger,
    )
    min_score = float(np.clip(min_score, 0.0, 1.0))
    normalized_prompt_labels = [
        normalize_detector_label(prompt_text_to_label(prompt_text))
        for prompt_text in prompt_texts
    ]

    with torch.no_grad():
        prompt_tokens = clip.tokenize(prompt_texts).to(device)
        prompt_features = model.encode_text(prompt_tokens).float()
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

    image_flags: list[list[dict]] = []
    valid_paths: list[Path] = []

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            cached = feature_cache.get(image_path) if feature_cache is not None else None
            image = cached.image if cached is not None else load_rgb_image(image_path)
            if image is None:
                logger.warning("Skipping unreadable image during detailed scene CLIP labeling: %s", image_path)
                continue

            candidates = generate_scene_region_candidates(image, prompt_texts)
            if not candidates:
                image_flags.append([])
                valid_paths.append(image_path)
                continue

            crops: list[Image.Image] = []
            for candidate in candidates:
                region = candidate["region"]
                crops.append(
                    image.crop(
                        (
                            int(region["left"]),
                            int(region["top"]),
                            int(region["right"]),
                            int(region["bottom"]),
                        )
                    )
                )

            crop_tensor = torch.stack([preprocess(crop) for crop in crops]).to(device)
            crop_features = model.encode_image(crop_tensor).float()
            crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
            similarity = (crop_features @ prompt_features.T).cpu().numpy().astype(np.float32, copy=False)

            raw_flags: list[dict] = []
            for candidate_index, candidate in enumerate(candidates):
                candidate_labels = set(candidate["prompt_labels"])
                heuristic_scores = {
                    normalize_detector_label(label): float(np.clip(score, 0.0, 1.0))
                    for label, score in dict(candidate.get("heuristic_scores", {})).items()
                }
                region = dict(candidate["region"])
                candidate_source = str(region.get("scene_candidate_source", "heuristic"))
                for prompt_index, normalized_label in enumerate(normalized_prompt_labels):
                    if not normalized_label or normalized_label not in candidate_labels:
                        continue
                    clip_score = float(np.clip(similarity[candidate_index, prompt_index], 0.0, 1.0))
                    heuristic_score = float(heuristic_scores.get(normalized_label, 0.0))
                    combined_score = max(clip_score, heuristic_score)
                    if combined_score < scene_label_score_floor(normalized_label, min_score):
                        continue
                    region_payload = dict(region)
                    region_payload["source"] = "scene_clip" if clip_score >= heuristic_score else "scene_heuristic"
                    region_payload["scene_candidate_source"] = candidate_source
                    region_payload["clip_similarity"] = similarity_to_percent(clip_score)
                    region_payload["prompt_region_score_percent"] = similarity_to_percent(combined_score)
                    raw_flags.append(
                        {
                            "label": prompt_text_to_label(prompt_texts[prompt_index]) or prompt_texts[prompt_index],
                            "prompt": prompt_texts[prompt_index],
                            "score_percent": similarity_to_percent(combined_score),
                            "region": region_payload,
                        }
                    )

            raw_flags.sort(
                key=lambda item: (
                    scene_candidate_source_priority(str(item.get("region", {}).get("scene_candidate_source", ""))),
                    -float(item.get("score_percent", 0.0)),
                    region_area(item.get("region")),
                )
            )

            kept_flags: list[dict] = []
            label_counts: dict[str, int] = {}
            for flag in raw_flags:
                normalized_label = normalize_detector_label(flag["label"])
                if label_counts.get(normalized_label, 0) >= scene_duplicate_limit(normalized_label):
                    continue
                overlaps_existing = False
                for existing_flag in kept_flags:
                    if normalize_detector_label(existing_flag["label"]) != normalized_label:
                        continue
                    if region_iou(flag.get("region"), existing_flag.get("region")) >= 0.55:
                        overlaps_existing = True
                        break
                if overlaps_existing:
                    continue
                kept_flags.append(flag)
                label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1

            image_flags.append(kept_flags)
            valid_paths.append(image_path)

            if index % 8 == 0 or index == len(image_paths):
                logger.info("Scored detailed heuristic scene CLIP labels for %s/%s images", index, len(image_paths))

    return image_flags, valid_paths


def build_yolo_scene_clip_image_flag_payload(
    image_paths: list[Path],
    prompt_texts: list[str],
    prompt_set: str,
    *,
    model_name: str,
    device: str,
    cache_dir: Path,
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    yolo_model_path: str,
    yolo_confidence_threshold: float,
    yolo_iou_threshold: float,
    yolo_image_size: int,
    yolo_max_detections: int,
    yolo_retina_masks: bool,
    top_k: int,
    min_score: float,
    scene_clip_min_score: float,
    include_labels: set[str] | None = None,
) -> dict:
    top_k = max(1, int(top_k))
    min_score = float(np.clip(min_score, 0.0, 1.0))
    normalized_include_labels = (
        {prompt_text_to_label(label).lower() for label in include_labels if prompt_text_to_label(label)}
        if include_labels
        else None
    )

    object_prompt_indices = select_prompt_indices_for_labels(prompt_texts, YOLO_SCENE_CLIP_OBJECT_LABELS)
    object_prompt_texts = [prompt_texts[prompt_index] for prompt_index in object_prompt_indices]
    object_scores = np.empty((0, 0), dtype=np.float32)
    object_regions: list[list[dict | None]] = []
    object_paths: list[Path] = []
    if object_prompt_texts:
        object_scores, object_regions, object_paths = extract_yolo_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=object_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence_threshold,
            iou_threshold=yolo_iou_threshold,
            image_size=yolo_image_size,
            max_detections=yolo_max_detections,
            retina_masks=yolo_retina_masks,
        )

    scene_prompt_indices = select_prompt_indices_for_labels(prompt_texts, YOLO_SCENE_CLIP_SCENE_LABELS)
    scene_prompt_texts = [prompt_texts[prompt_index] for prompt_index in scene_prompt_indices]
    scene_flags_by_path: dict[Path, list[dict]] = {}
    if scene_prompt_texts:
        scene_flag_rows, scene_paths = extract_scene_clip_flag_items(
            image_paths=image_paths,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=scene_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            min_score=scene_clip_min_score,
        )
        scene_flags_by_path = {
            scene_path: list(flags)
            for scene_path, flags in zip(scene_paths, scene_flag_rows)
        }

    object_flags_by_path: dict[Path, list[dict]] = {}
    object_score_floor = max(min_score, min(1.0, float(yolo_confidence_threshold)))
    for row_index, image_path in enumerate(object_paths):
        flags: list[dict] = []
        for prompt_index, prompt_text in enumerate(object_prompt_texts):
            score = float(object_scores[row_index, prompt_index]) if object_scores.size else 0.0
            if score < object_score_floor:
                continue
            region = object_regions[row_index][prompt_index] if row_index < len(object_regions) and prompt_index < len(object_regions[row_index]) else None
            label = prompt_text_to_label(prompt_text) or prompt_text
            if normalized_include_labels is not None and label.lower() not in normalized_include_labels:
                continue
            flag_entry = {
                "label": label,
                "prompt": prompt_text,
                "score_percent": similarity_to_percent(score),
            }
            if region is not None:
                flag_entry["region"] = dict(region)
            flags.append(flag_entry)
        object_flags_by_path[image_path] = flags

    images_payload: list[dict] = []
    for image_path in image_paths:
        merged_flags = list(object_flags_by_path.get(image_path, [])) + list(scene_flags_by_path.get(image_path, []))
        if normalized_include_labels is not None:
            merged_flags = [
                flag for flag in merged_flags
                if normalize_detector_label(flag.get("label", "")) in normalized_include_labels
            ]
        merged_flags.sort(
            key=lambda item: (
                0 if normalize_detector_label(str(item.get("region", {}).get("source", ""))) == "yolo" else 1,
                scene_candidate_source_priority(str(item.get("region", {}).get("scene_candidate_source", ""))),
                -float(item.get("score_percent", 0.0)),
                region_area(item.get("region")),
            )
        )
        images_payload.append(
            {
                "image": image_path.name,
                "flagged_items": merged_flags[:top_k],
            }
        )

    return {
        "prompt_set": prompt_set,
        "top_k": top_k,
        "min_score_percent": similarity_to_percent(min_score),
        "images": images_payload,
    }


def extract_open_vocab_hybrid_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    open_vocab_model_id: str,
    open_vocab_score_threshold: float,
    segmentation_min_area_ratio: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    backend_outputs: list[tuple[np.ndarray, list[list[dict | None]] | None, list[Path]]] = []
    resolved_paths = list(image_paths)

    open_vocab_prompt_indices = [
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if (
            prompt_text_to_label(prompt_text).lower() not in HYBRID_CLIP_LABELS
            and prompt_text_to_label(prompt_text).lower() not in SCENE_FLAG_LABELS
        )
    ]
    if open_vocab_prompt_indices:
        open_vocab_prompt_texts = [prompt_texts[prompt_index] for prompt_index in open_vocab_prompt_indices]
        open_vocab_scores_subset, open_vocab_regions_subset, open_vocab_paths = extract_open_vocab_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=open_vocab_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_id=open_vocab_model_id,
            score_threshold=open_vocab_score_threshold,
        )
        open_vocab_scores, open_vocab_regions = expand_prompt_backend_output(
            scores_subset=open_vocab_scores_subset,
            regions_subset=open_vocab_regions_subset,
            subset_prompt_indices=open_vocab_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=open_vocab_paths,
        )
        backend_outputs.append((open_vocab_scores, open_vocab_regions, open_vocab_paths))
        resolved_paths = open_vocab_paths

    _, _, segformer_label_to_id = load_segformer_ade20k_runtime(device=device)
    segformer_class_names = {int(class_id): str(label) for label, class_id in segformer_label_to_id.items()}
    segformer_prompt_lookup = build_segformer_prompt_class_lookup(prompt_texts, segformer_class_names)
    segformer_prompt_indices = prompt_indices_from_class_lookup(segformer_prompt_lookup)
    if segformer_prompt_indices:
        seg_scores, seg_regions, seg_paths = extract_segmentation_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            min_area_ratio=segmentation_min_area_ratio,
        )
        backend_outputs.append((seg_scores, seg_regions, seg_paths))
        resolved_paths = seg_paths

    clip_prompt_indices = {
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if (
            prompt_index not in open_vocab_prompt_indices
            and prompt_index not in segformer_prompt_indices
        )
        or prompt_text_to_label(prompt_text).lower() in HYBRID_CLIP_LABELS
    }
    if clip_prompt_indices:
        clip_prompt_indices = set(sorted(clip_prompt_indices))
        expanded_clip_prompts: list[str] = []
        expanded_prompt_targets: list[int | None] = []
        requested_scene_labels = {
            prompt_text_to_label(prompt_texts[prompt_index]).lower(): prompt_index for prompt_index in sorted(clip_prompt_indices)
        }
        for prompt_index in sorted(clip_prompt_indices):
            prompt_text = prompt_texts[prompt_index]
            normalized_label = prompt_text_to_label(prompt_text).lower()
            if normalized_label == "outdoor":
                expanded_clip_prompts.append("an outdoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "indoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an indoor scene")
                    expanded_prompt_targets.append(None)
                continue
            if normalized_label == "indoor":
                expanded_clip_prompts.append("an indoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "outdoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an outdoor scene")
                    expanded_prompt_targets.append(None)
                continue
            expanded_clip_prompts.append(prompt_text)
            expanded_prompt_targets.append(prompt_index)

        clip_scores_subset, clip_regions_subset, clip_paths = extract_clip_flag_scores(
            image_paths=clip_fallback_paths,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=expanded_clip_prompts,
            feature_cache=feature_cache,
            logger=logger,
        )
        full_clip_scores = np.zeros((len(clip_paths), len(prompt_texts)), dtype=np.float32)
        full_clip_regions: list[list[dict | None]] = [[None for _ in prompt_texts] for _ in clip_paths]
        for subset_index, prompt_index in enumerate(expanded_prompt_targets):
            if prompt_index is None:
                continue
            if clip_scores_subset.size:
                full_clip_scores[:, prompt_index] = clip_scores_subset[:, subset_index]
            if clip_regions_subset:
                for image_index in range(min(len(clip_paths), len(clip_regions_subset))):
                    region = clip_regions_subset[image_index][subset_index]
                    if region is not None:
                        full_clip_regions[image_index][prompt_index] = dict(region)
        backend_outputs.append((full_clip_scores, full_clip_regions, clip_paths))

    merged_scores, merged_regions = merge_flag_backend_outputs(
        image_paths=image_paths,
        prompt_count=len(prompt_texts),
        backend_outputs=backend_outputs,
    )
    return merged_scores, merged_regions, list(image_paths)


def extract_advanced_hybrid_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    open_vocab_model_id: str,
    open_vocab_score_threshold: float,
    grounding_dino_model_id: str,
    grounding_dino_box_threshold: float,
    grounding_dino_text_threshold: float,
    segmentation_min_area_ratio: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    backend_outputs: list[tuple[np.ndarray, list[list[dict | None]] | None, list[Path]]] = []

    detector_prompt_indices = [
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if prompt_text_to_label(prompt_text).lower() not in HYBRID_CLIP_LABELS
    ]
    if detector_prompt_indices:
        detector_prompt_texts = [prompt_texts[prompt_index] for prompt_index in detector_prompt_indices]

        grounding_dino_scores_subset, grounding_dino_regions_subset, grounding_dino_paths = extract_grounding_dino_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=detector_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_id=grounding_dino_model_id,
            box_threshold=grounding_dino_box_threshold,
            text_threshold=grounding_dino_text_threshold,
        )
        grounding_dino_scores, grounding_dino_regions = expand_prompt_backend_output(
            scores_subset=grounding_dino_scores_subset,
            regions_subset=grounding_dino_regions_subset,
            subset_prompt_indices=detector_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=grounding_dino_paths,
        )
        backend_outputs.append((grounding_dino_scores, grounding_dino_regions, grounding_dino_paths))

        open_vocab_scores_subset, open_vocab_regions_subset, open_vocab_paths = extract_open_vocab_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=detector_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_id=open_vocab_model_id,
            score_threshold=open_vocab_score_threshold,
        )
        open_vocab_scores, open_vocab_regions = expand_prompt_backend_output(
            scores_subset=open_vocab_scores_subset,
            regions_subset=open_vocab_regions_subset,
            subset_prompt_indices=detector_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=open_vocab_paths,
        )
        backend_outputs.append((open_vocab_scores, open_vocab_regions, open_vocab_paths))

    _, _, segformer_label_to_id = load_segformer_ade20k_runtime(device=device)
    segformer_class_names = {int(class_id): str(label) for label, class_id in segformer_label_to_id.items()}
    segformer_prompt_lookup = build_segformer_prompt_class_lookup(prompt_texts, segformer_class_names)
    segformer_prompt_indices = prompt_indices_from_class_lookup(segformer_prompt_lookup)
    if segformer_prompt_indices:
        seg_scores, seg_regions, seg_paths = extract_segmentation_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            min_area_ratio=segmentation_min_area_ratio,
        )
        backend_outputs.append((seg_scores, seg_regions, seg_paths))

    clip_prompt_indices = {
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if (
            prompt_index not in detector_prompt_indices
            and prompt_index not in segformer_prompt_indices
        )
        or prompt_text_to_label(prompt_text).lower() in HYBRID_CLIP_LABELS
    }
    if clip_prompt_indices:
        clip_prompt_indices = set(sorted(clip_prompt_indices))
        expanded_clip_prompts: list[str] = []
        expanded_prompt_targets: list[int | None] = []
        requested_scene_labels = {
            prompt_text_to_label(prompt_texts[prompt_index]).lower(): prompt_index for prompt_index in sorted(clip_prompt_indices)
        }
        for prompt_index in sorted(clip_prompt_indices):
            prompt_text = prompt_texts[prompt_index]
            normalized_label = prompt_text_to_label(prompt_text).lower()
            if normalized_label == "outdoor":
                expanded_clip_prompts.append("an outdoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "indoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an indoor scene")
                    expanded_prompt_targets.append(None)
                continue
            if normalized_label == "indoor":
                expanded_clip_prompts.append("an indoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "outdoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an outdoor scene")
                    expanded_prompt_targets.append(None)
                continue
            expanded_clip_prompts.append(prompt_text)
            expanded_prompt_targets.append(prompt_index)

        clip_scores_subset, clip_regions_subset, clip_paths = extract_clip_flag_scores(
            image_paths=image_paths,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=expanded_clip_prompts,
            feature_cache=feature_cache,
            logger=logger,
        )
        full_clip_scores = np.zeros((len(clip_paths), len(prompt_texts)), dtype=np.float32)
        full_clip_regions: list[list[dict | None]] = [[None for _ in prompt_texts] for _ in clip_paths]
        for subset_index, prompt_index in enumerate(expanded_prompt_targets):
            if prompt_index is None:
                continue
            if clip_scores_subset.size:
                full_clip_scores[:, prompt_index] = clip_scores_subset[:, subset_index]
            if clip_regions_subset:
                for image_index in range(min(len(clip_paths), len(clip_regions_subset))):
                    region = clip_regions_subset[image_index][subset_index]
                    if region is not None:
                        full_clip_regions[image_index][prompt_index] = dict(region)
        backend_outputs.append((full_clip_scores, full_clip_regions, clip_paths))

    merged_scores, merged_regions = merge_flag_backend_outputs(
        image_paths=image_paths,
        prompt_count=len(prompt_texts),
        backend_outputs=backend_outputs,
    )
    return merged_scores, merged_regions, list(image_paths)


def extract_grounding_dino_hybrid_flag_scores(
    image_paths: list[Path],
    model_name: str,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    grounding_dino_model_id: str,
    grounding_dino_box_threshold: float,
    grounding_dino_text_threshold: float,
    segmentation_min_area_ratio: float,
) -> tuple[np.ndarray, list[list[dict | None]], list[Path]]:
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), [], []

    backend_outputs: list[tuple[np.ndarray, list[list[dict | None]] | None, list[Path]]] = []
    resolved_paths = list(image_paths)

    grounding_dino_prompt_indices = [
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if prompt_text_to_label(prompt_text).lower() not in HYBRID_CLIP_LABELS
    ]
    if grounding_dino_prompt_indices:
        grounding_dino_prompt_texts = [prompt_texts[prompt_index] for prompt_index in grounding_dino_prompt_indices]
        grounding_dino_scores_subset, grounding_dino_regions_subset, grounding_dino_paths = extract_grounding_dino_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=grounding_dino_prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_id=grounding_dino_model_id,
            box_threshold=grounding_dino_box_threshold,
            text_threshold=grounding_dino_text_threshold,
        )
        grounding_dino_scores, grounding_dino_regions = expand_prompt_backend_output(
            scores_subset=grounding_dino_scores_subset,
            regions_subset=grounding_dino_regions_subset,
            subset_prompt_indices=grounding_dino_prompt_indices,
            prompt_count=len(prompt_texts),
            resolved_paths=grounding_dino_paths,
        )
        backend_outputs.append((grounding_dino_scores, grounding_dino_regions, grounding_dino_paths))
        resolved_paths = grounding_dino_paths

    _, _, segformer_label_to_id = load_segformer_ade20k_runtime(device=device)
    segformer_class_names = {int(class_id): str(label) for label, class_id in segformer_label_to_id.items()}
    segformer_prompt_lookup = build_segformer_prompt_class_lookup(prompt_texts, segformer_class_names)
    segformer_prompt_indices = prompt_indices_from_class_lookup(segformer_prompt_lookup)
    if segformer_prompt_indices:
        seg_scores, seg_regions, seg_paths = extract_segmentation_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            min_area_ratio=segmentation_min_area_ratio,
        )
        backend_outputs.append((seg_scores, seg_regions, seg_paths))
        resolved_paths = seg_paths

    clip_prompt_indices = {
        prompt_index
        for prompt_index, prompt_text in enumerate(prompt_texts)
        if (
            prompt_index not in grounding_dino_prompt_indices
            and prompt_index not in segformer_prompt_indices
        )
        or prompt_text_to_label(prompt_text).lower() in HYBRID_CLIP_LABELS
    }
    if clip_prompt_indices:
        clip_prompt_indices = set(sorted(clip_prompt_indices))
        expanded_clip_prompts: list[str] = []
        expanded_prompt_targets: list[int | None] = []
        requested_scene_labels = {
            prompt_text_to_label(prompt_texts[prompt_index]).lower(): prompt_index for prompt_index in sorted(clip_prompt_indices)
        }
        for prompt_index in sorted(clip_prompt_indices):
            prompt_text = prompt_texts[prompt_index]
            normalized_label = prompt_text_to_label(prompt_text).lower()
            if normalized_label == "outdoor":
                expanded_clip_prompts.append("an outdoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "indoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an indoor scene")
                    expanded_prompt_targets.append(None)
                continue
            if normalized_label == "indoor":
                expanded_clip_prompts.append("an indoor scene")
                expanded_prompt_targets.append(prompt_index)
                if "outdoor" not in requested_scene_labels:
                    expanded_clip_prompts.append("an outdoor scene")
                    expanded_prompt_targets.append(None)
                continue
            expanded_clip_prompts.append(prompt_text)
            expanded_prompt_targets.append(prompt_index)

        clip_scores_subset, clip_regions_subset, clip_paths = extract_clip_flag_scores(
            image_paths=image_paths,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=expanded_clip_prompts,
            feature_cache=feature_cache,
            logger=logger,
        )
        full_clip_scores = np.zeros((len(clip_paths), len(prompt_texts)), dtype=np.float32)
        full_clip_regions: list[list[dict | None]] = [[None for _ in prompt_texts] for _ in clip_paths]
        for subset_index, prompt_index in enumerate(expanded_prompt_targets):
            if prompt_index is None:
                continue
            if clip_scores_subset.size:
                full_clip_scores[:, prompt_index] = clip_scores_subset[:, subset_index]
            if clip_regions_subset:
                for image_index in range(min(len(clip_paths), len(clip_regions_subset))):
                    region = clip_regions_subset[image_index][subset_index]
                    if region is not None:
                        full_clip_regions[image_index][prompt_index] = dict(region)
        backend_outputs.append((full_clip_scores, full_clip_regions, clip_paths))
        resolved_paths = clip_paths

    merged_scores, merged_regions = merge_flag_backend_outputs(
        image_paths=image_paths,
        prompt_count=len(prompt_texts),
        backend_outputs=backend_outputs,
    )
    return merged_scores, merged_regions, resolved_paths


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
    image_flag_lookup: dict[str, list[dict]] | None,
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
                "flagged_items": image_flag_lookup.get(image_path.name, []) if image_flag_lookup is not None else [],
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


def measure_text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    if not text:
        return 0, 0
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return max(0, right - left), max(0, bottom - top)


def load_annotation_font(font_size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidate_names = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf"]
    )
    for font_name in candidate_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def truncate_text_to_width(
    text: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    normalized = " ".join(str(text).split())
    if not normalized or max_width <= 0:
        return ""

    if measure_text_size(draw, normalized, font)[0] <= max_width:
        return normalized

    ellipsis = "..."
    candidate = normalized
    while candidate:
        candidate = candidate[:-1].rstrip()
        shortened = f"{candidate}{ellipsis}" if candidate else ellipsis
        if measure_text_size(draw, shortened, font)[0] <= max_width:
            return shortened
    return ellipsis


def boundary_bounds(boundary_points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not boundary_points:
        return None
    xs = [point[0] for point in boundary_points]
    ys = [point[1] for point in boundary_points]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def is_rectangular_boundary(
    boundary_points: list[tuple[int, int]],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> bool:
    if len(boundary_points) != 4:
        return False
    expected = {
        (int(left), int(top)),
        (int(max(left, right - 1)), int(top)),
        (int(max(left, right - 1)), int(max(top, bottom - 1))),
        (int(left), int(max(top, bottom - 1))),
    }
    return set(boundary_points) == expected


def refine_box_boundary_with_grabcut(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> list[tuple[int, int]] | None:
    box_width = max(1, int(right - left))
    box_height = max(1, int(bottom - top))
    if box_width < 24 or box_height < 24:
        return None

    image_array = np.asarray(image.convert("RGB"))
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        return None

    image_height, image_width = image_array.shape[:2]
    pad_x = max(6, int(round(box_width * 0.08)))
    pad_y = max(6, int(round(box_height * 0.08)))
    rect_left = max(0, left - pad_x)
    rect_top = max(0, top - pad_y)
    rect_right = min(image_width, right + pad_x)
    rect_bottom = min(image_height, bottom + pad_y)
    rect_width = int(rect_right - rect_left)
    rect_height = int(rect_bottom - rect_top)
    if rect_width < 4 or rect_height < 4:
        return None

    mask = np.full((image_height, image_width), cv2.GC_BGD, dtype=np.uint8)
    mask[rect_top:rect_bottom, rect_left:rect_right] = cv2.GC_PR_FGD

    inner_pad_x = max(2, int(round(box_width * 0.18)))
    inner_pad_y = max(2, int(round(box_height * 0.18)))
    inner_left = min(max(left + inner_pad_x, rect_left), max(rect_left, right - 2))
    inner_top = min(max(top + inner_pad_y, rect_top), max(rect_top, bottom - 2))
    inner_right = max(inner_left + 1, min(right - inner_pad_x, rect_right))
    inner_bottom = max(inner_top + 1, min(bottom - inner_pad_y, rect_bottom))
    if inner_right > inner_left and inner_bottom > inner_top:
        mask[inner_top:inner_bottom, inner_left:inner_right] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR),
            mask,
            None,
            bgd_model,
            fgd_model,
            4,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return None

    foreground_mask = np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD).astype(np.uint8)
    foreground_mask[:rect_top, :] = 0
    foreground_mask[rect_bottom:, :] = 0
    foreground_mask[:, :rect_left] = 0
    foreground_mask[:, rect_right:] = 0

    kernel_size = max(3, int(round(min(box_width, box_height) * 0.03)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    region, component_mask, component_area = largest_component_region(foreground_mask.astype(bool))
    if region is None or component_mask is None or component_area <= 0:
        return None

    box_area = float(box_width * box_height)
    rect_area = float(rect_width * rect_height)
    area_ratio = float(component_area / max(1.0, rect_area))
    if component_area < box_area * 0.05 or area_ratio > 0.92:
        return None

    boundary = mask_to_boundary_points(component_mask)
    if not boundary:
        return None

    return [(int(point["x"]), int(point["y"])) for point in boundary]


def annotate_image_with_flags(image: Image.Image, flagged_items: list[dict] | None) -> Image.Image:
    annotations: list[tuple[str, tuple[int, int, int, int], list[tuple[int, int]], str, int]] = []
    for item in flagged_items or []:
        label = " ".join(str(item.get("label", "")).split())
        region = item.get("region")
        if not label or not isinstance(region, dict):
            continue
        try:
            left = int(round(float(region.get("left", 0))))
            top = int(round(float(region.get("top", 0))))
            right = int(round(float(region.get("right", 0))))
            bottom = int(round(float(region.get("bottom", 0))))
        except (TypeError, ValueError):
            continue
        left = max(0, min(left, image.width - 1))
        top = max(0, min(top, image.height - 1))
        right = max(left + 1, min(right, image.width))
        bottom = max(top + 1, min(bottom, image.height))
        source = str(region.get("source", "")).strip().lower()
        boundary_points = coerce_boundary_points(region, image.width, image.height)
        if source in {"yolo", "open_vocab", "grounding_dino"} and (
            not boundary_points or is_rectangular_boundary(boundary_points, left, top, right, bottom)
        ):
            refined_boundary = refine_box_boundary_with_grabcut(image, left, top, right, bottom)
            if refined_boundary:
                boundary_points = refined_boundary
                refined_bounds = boundary_bounds(boundary_points)
                if refined_bounds is not None:
                    left, top, right, bottom = refined_bounds
        area = max(1, int(max(1, right - left) * max(1, bottom - top)))
        annotations.append((label, (left, top, right, bottom), boundary_points, source, area))

    if not annotations:
        return image.copy()

    annotations.sort(key=lambda item: item[4], reverse=True)

    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    min_dimension = max(1, min(annotated.width, annotated.height))
    outline_width = max(2, int(min_dimension * 0.0035))
    badge_padding_x = max(10, int(min_dimension * 0.012))
    badge_padding_y = max(6, int(min_dimension * 0.008))
    badge_gap = max(4, int(min_dimension * 0.006))
    label_font = load_annotation_font(max(18, int(min_dimension * 0.026)), bold=True)
    palette = [
        (239, 68, 68),
        (34, 197, 94),
        (59, 130, 246),
        (249, 115, 22),
        (168, 85, 247),
        (236, 72, 153),
    ]
    occupied_badges: dict[tuple[int, int], int] = {}

    for index, (label, (left, top, right, bottom), boundary_points, source, _area) in enumerate(annotations):
        color = palette[index % len(palette)]
        if len(boundary_points) >= 2:
            path = boundary_points + ([boundary_points[0]] if len(boundary_points) >= 3 else [])
            if len(boundary_points) >= 3:
                fill_alpha = 28 if source in {"segformer_ade20k"} else 18
                draw.polygon(boundary_points, fill=(*color, fill_alpha))
            draw.line(path, fill=(*color, 228), width=outline_width)
        else:
            draw.rectangle(
                (left, top, right, bottom),
                outline=(*color, 228),
                width=outline_width,
            )

        max_label_width = max(
            20,
            min(
                annotated.width - (2 * badge_padding_x) - 12,
                max(right - left, int(annotated.width * 0.35)),
            ),
        )
        text = truncate_text_to_width(label, draw, label_font, max_label_width)
        text_width, text_height = measure_text_size(draw, text, label_font)
        badge_width = text_width + (2 * badge_padding_x)
        badge_height = text_height + (2 * badge_padding_y)
        badge_left = max(6, min(left, annotated.width - badge_width - 6))
        preferred_top = top - badge_height - badge_gap
        if preferred_top < 6:
            preferred_top = min(annotated.height - badge_height - 6, top + badge_gap)
        badge_key = (badge_left // 24, preferred_top // 24)
        badge_offset = occupied_badges.get(badge_key, 0)
        occupied_badges[badge_key] = badge_offset + 1
        badge_top = min(annotated.height - badge_height - 6, preferred_top + badge_offset * (badge_height + 4))
        badge_box = (badge_left, badge_top, badge_left + badge_width, badge_top + badge_height)
        draw.rounded_rectangle(
            badge_box,
            radius=max(8, badge_height // 3),
            fill=(*color, 236),
            outline=(255, 255, 255, 230),
            width=max(1, outline_width // 2),
        )
        draw.text(
            (badge_left + badge_padding_x, badge_top + badge_padding_y - 1),
            text,
            font=label_font,
            fill=(255, 255, 255, 255),
        )

    return Image.alpha_composite(annotated, overlay).convert("RGB")


def save_exported_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, int | bool] = {}
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 92, "optimize": True, "subsampling": 0}
    elif suffix == ".png":
        save_kwargs = {"compress_level": 6}
    image.save(output_path, **save_kwargs)


def export_cluster_image(
    image_path: Path,
    destination_path: Path,
    flagged_items: list[dict] | None,
    annotate_flagged_images: bool,
    feature_cache: dict[Path, CachedImageFeatures] | None = None,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not annotate_flagged_images:
        shutil.copy2(image_path, destination_path)
        return

    cached = feature_cache.get(image_path) if feature_cache is not None else None
    source_image = cached.image.copy() if cached is not None else load_rgb_image(image_path)
    if source_image is None:
        shutil.copy2(image_path, destination_path)
        return

    save_exported_image(annotate_image_with_flags(source_image, flagged_items), destination_path)


def resolve_annotate_flagged_images(args: argparse.Namespace) -> bool:
    explicit_value = getattr(args, "annotate_flagged_images", None)
    if explicit_value is None:
        return bool(args.flag_items)
    return bool(explicit_value)


def build_resolved_run_settings(
    args: argparse.Namespace,
    input_dir: Path,
    requested_output_dir: Path,
    output_dir: Path,
    resolved_device: str,
    annotate_flagged_images: bool,
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
        "flag_items": bool(args.flag_items),
        "flag_prompt_set": args.flag_prompt_set,
        "flag_detector": args.flag_detector,
        "flag_scoring_mode": (
            "yolov8n_seg_scene_clip_heuristics"
            if args.flag_items and args.flag_detector == "yolo_scene_clip"
            else "hybrid_yolov8_owlv2_segformer_clip"
            if args.flag_items and args.flag_detector == "hybrid"
            else "sam_deeplab_yolov8_clip"
            if args.flag_items and args.flag_detector == "sam_deeplab_yolo_clip"
            else "hybrid_owlv2_segformer_clip"
            if args.flag_items and args.flag_detector == "open_vocab_hybrid"
            else "open_vocabulary_detection"
            if args.flag_items and args.flag_detector == "open_vocab"
            else "yolov8_detection"
            if args.flag_items and args.flag_detector == "yolo"
            else "whole_image_plus_tiles_max"
            if args.flag_items and args.flag_detector == "clip"
            else "semantic_segmentation_ade20k"
            if args.flag_items and args.flag_detector == "segformer_ade20k"
            else "disabled"
        ),
        "yolo_model": str(args.yolo_model),
        "yolo_confidence": float(args.yolo_confidence),
        "yolo_iou": float(args.yolo_iou),
        "yolo_imgsz": int(args.yolo_imgsz),
        "yolo_max_det": int(args.yolo_max_det),
        "yolo_retina_masks": bool(args.yolo_retina_masks),
        "sam_model": str(args.sam_model),
        "deeplab_model": str(args.deeplab_model),
        "deeplab_min_area": float(args.deeplab_min_area),
        "open_vocab_model": str(args.open_vocab_model),
        "open_vocab_threshold": float(args.open_vocab_threshold),
        "flag_top_k": max(1, int(args.flag_top_k)),
        "flag_min_score": float(args.flag_min_score),
        "scene_clip_min_score": float(args.scene_clip_min_score),
        "flag_include_labels": [str(label) for label in (args.flag_include_labels or [])],
        "segmentation_min_area": float(args.segmentation_min_area),
        "annotate_flagged_images": bool(annotate_flagged_images),
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


def prompt_text_to_label(prompt_text: str) -> str:
    label = " ".join(str(prompt_text).strip().split())
    if not label:
        return ""

    prefixes = (
        "a real estate interior photo of ",
        "a real estate exterior photo of ",
        "a real estate photo of ",
        "an interior photo of ",
        "an exterior photo of ",
        "an interior scene with ",
        "an exterior scene with ",
        "an ",
        "a ",
        "the ",
    )
    changed = True
    while changed and label:
        lowered = label.lower()
        changed = False
        for prefix in prefixes:
            if lowered.startswith(prefix):
                label = label[len(prefix) :].strip()
                changed = True
                break

    return label.strip(" .")


def normalize_detector_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def build_open_vocab_prompt_variants(
    prompt_texts: list[str],
) -> tuple[list[str], list[int], dict[str, list[int]]]:
    expanded_prompt_texts: list[str] = []
    expanded_prompt_targets: list[int] = []
    prompt_label_to_indices: dict[str, list[int]] = {}

    for prompt_index, prompt_text in enumerate(prompt_texts):
        label = prompt_text_to_label(prompt_text) or prompt_text
        normalized_label = normalize_detector_label(label)
        candidate_texts = [str(prompt_text).strip()]
        if normalized_label:
            candidate_texts.append(f"a photo of {normalized_label}")
        for alias in OPEN_VOCAB_LABEL_ALIASES.get(normalized_label, ()):
            normalized_alias = normalize_detector_label(alias)
            if not normalized_alias:
                continue
            candidate_texts.append(normalized_alias)
            candidate_texts.append(f"a photo of {normalized_alias}")

        seen_variant_texts: set[str] = set()
        variant_count = 0
        for candidate_text in candidate_texts:
            normalized_candidate_text = normalize_detector_label(candidate_text)
            if not normalized_candidate_text or normalized_candidate_text in seen_variant_texts:
                continue
            seen_variant_texts.add(normalized_candidate_text)
            expanded_prompt_texts.append(candidate_text)
            expanded_prompt_targets.append(prompt_index)
            prompt_label = normalize_detector_label(prompt_text_to_label(candidate_text) or candidate_text)
            if prompt_label:
                prompt_label_to_indices.setdefault(prompt_label, []).append(prompt_index)
            variant_count += 1
            if variant_count >= 4:
                break

    return expanded_prompt_texts, expanded_prompt_targets, prompt_label_to_indices


def build_grounding_dino_prompt_variants(
    prompt_texts: list[str],
) -> tuple[list[str], list[int], dict[str, list[int]]]:
    expanded_prompt_texts: list[str] = []
    expanded_prompt_targets: list[int] = []
    prompt_label_to_indices: dict[str, list[int]] = {}

    for prompt_index, prompt_text in enumerate(prompt_texts):
        label = normalize_detector_label(prompt_text_to_label(prompt_text) or prompt_text)
        if not label:
            continue

        candidate_texts: list[str] = [label]
        for alias in OPEN_VOCAB_LABEL_ALIASES.get(label, ()):
            normalized_alias = normalize_detector_label(alias)
            if normalized_alias:
                candidate_texts.append(normalized_alias)

        seen_variant_texts: set[str] = set()
        variant_count = 0
        for candidate_text in candidate_texts:
            normalized_candidate_text = normalize_detector_label(candidate_text)
            if not normalized_candidate_text or normalized_candidate_text in seen_variant_texts:
                continue
            seen_variant_texts.add(normalized_candidate_text)
            expanded_prompt_texts.append(normalized_candidate_text)
            expanded_prompt_targets.append(prompt_index)
            prompt_label_to_indices.setdefault(normalized_candidate_text, []).append(prompt_index)
            stripped_label = normalize_detector_label(prompt_text_to_label(normalized_candidate_text) or normalized_candidate_text)
            if stripped_label:
                prompt_label_to_indices.setdefault(stripped_label, []).append(prompt_index)
            variant_count += 1
            if variant_count >= 4:
                break

    return expanded_prompt_texts, expanded_prompt_targets, prompt_label_to_indices


def detector_label_aliases(label: str, alias_map: dict[str, tuple[str, ...]] | None = None) -> set[str]:
    normalized = normalize_detector_label(label)
    if not normalized:
        return set()

    aliases = {normalized}
    if normalized.endswith("s") and len(normalized) > 3:
        aliases.add(normalized[:-1])
    elif not normalized.endswith("s"):
        aliases.add(f"{normalized}s")

    for alias in (alias_map or {}).get(normalized, ()):
        normalized_alias = normalize_detector_label(alias)
        if not normalized_alias:
            continue
        aliases.add(normalized_alias)
        if normalized_alias.endswith("s") and len(normalized_alias) > 3:
            aliases.add(normalized_alias[:-1])
        elif not normalized_alias.endswith("s"):
            aliases.add(f"{normalized_alias}s")

    return aliases


def resolve_named_class_lookup(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, dict):
        resolved: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                resolved[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return resolved
    if isinstance(raw_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


def build_detector_prompt_class_lookup(
    prompt_texts: list[str],
    class_names: dict[int, str],
    alias_map: dict[str, tuple[str, ...]] | None = None,
) -> dict[int, list[int]]:
    normalized_class_names = {
        class_id: normalize_detector_label(class_name)
        for class_id, class_name in class_names.items()
        if normalize_detector_label(class_name)
    }
    prompt_class_lookup: dict[int, list[int]] = {}
    for prompt_index, prompt_text in enumerate(prompt_texts):
        label = prompt_text_to_label(prompt_text)
        aliases = detector_label_aliases(label, alias_map=alias_map)
        if not aliases:
            continue
        for class_id, normalized_class_name in normalized_class_names.items():
            if normalized_class_name in aliases:
                prompt_class_lookup.setdefault(class_id, []).append(prompt_index)
    return prompt_class_lookup


def build_yolo_prompt_class_lookup(prompt_texts: list[str], class_names: dict[int, str]) -> dict[int, list[int]]:
    return build_detector_prompt_class_lookup(prompt_texts, class_names, alias_map=YOLO_LABEL_ALIASES)


def build_segformer_prompt_class_lookup(prompt_texts: list[str], class_names: dict[int, str]) -> dict[int, list[int]]:
    return build_detector_prompt_class_lookup(prompt_texts, class_names, alias_map=SEGFORMER_LABEL_ALIASES)


def build_deeplab_prompt_class_lookup(prompt_texts: list[str], class_names: dict[int, str]) -> dict[int, list[int]]:
    return build_detector_prompt_class_lookup(prompt_texts, class_names, alias_map=DEEPLAB_LABEL_ALIASES)


def prompt_indices_from_class_lookup(prompt_class_lookup: dict[int, list[int]]) -> set[int]:
    return {prompt_index for prompt_indices in prompt_class_lookup.values() for prompt_index in prompt_indices}


def count_localized_non_scene_flags(
    scores_row: np.ndarray,
    regions_row: list[dict | None] | None,
    prompt_texts: list[str],
    *,
    min_score: float,
) -> int:
    localized_count = 0
    prompt_count = min(len(prompt_texts), int(scores_row.shape[0]))
    for prompt_index in range(prompt_count):
        score = float(scores_row[prompt_index])
        if score < min_score:
            continue

        label = prompt_text_to_label(prompt_texts[prompt_index]).lower()
        if label in SCENE_FLAG_LABELS:
            continue

        region = None
        if regions_row is not None and prompt_index < len(regions_row):
            region = regions_row[prompt_index]
        source = normalize_detector_label(region.get("source", "")) if isinstance(region, dict) else ""
        if source in LOCALIZED_FLAG_SOURCES:
            localized_count += 1
    return localized_count


def select_low_coverage_image_paths(
    image_paths: list[Path],
    scores: np.ndarray,
    regions: list[list[dict | None]] | None,
    prompt_texts: list[str],
    *,
    min_score: float,
    min_localized_non_scene: int,
) -> list[Path]:
    if not image_paths:
        return []
    if scores.size == 0 or len(scores) != len(image_paths):
        return list(image_paths)

    min_score = float(np.clip(min_score, 0.0, 1.0))
    min_localized_non_scene = max(0, int(min_localized_non_scene))
    selected_paths: list[Path] = []
    for image_index, image_path in enumerate(image_paths):
        regions_row = regions[image_index] if regions is not None and image_index < len(regions) else None
        localized_non_scene_count = count_localized_non_scene_flags(
            scores[image_index],
            regions_row,
            prompt_texts,
            min_score=min_score,
        )
        if localized_non_scene_count < min_localized_non_scene:
            selected_paths.append(image_path)
    return selected_paths


def refine_regions_with_sam(
    image_paths: list[Path],
    device: str,
    prompt_texts: list[str],
    prompt_scores: np.ndarray,
    prompt_regions: list[list[dict | None]],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
    model_path: str,
    min_score: float,
) -> list[list[dict | None]]:
    if not image_paths or prompt_scores.size == 0 or not prompt_regions:
        return prompt_regions

    sam_model = load_sam_runtime(model_path=model_path)
    predict_device = "0" if device == "cuda" else "cpu"
    min_score = float(np.clip(min_score, 0.0, 1.0))
    refined_regions: list[list[dict | None]] = [
        [dict(region) if region is not None else None for region in regions_row]
        for regions_row in prompt_regions
    ]

    for image_index, image_path in enumerate(image_paths):
        if image_index >= len(refined_regions) or image_index >= len(prompt_scores):
            continue
        cached = feature_cache.get(image_path) if feature_cache is not None else None
        image = cached.image if cached is not None else load_rgb_image(image_path)
        if image is None:
            logger.warning("Skipping unreadable image during SAM refinement: %s", image_path)
            continue

        prompt_indices: list[int] = []
        prompt_boxes: list[list[int]] = []
        for prompt_index in range(min(len(prompt_texts), prompt_scores.shape[1], len(refined_regions[image_index]))):
            score = float(prompt_scores[image_index][prompt_index])
            region = refined_regions[image_index][prompt_index]
            if region is None or score < min_score:
                continue

            label = prompt_text_to_label(prompt_texts[prompt_index]).lower()
            if label == "outdoor":
                continue

            try:
                left = int(region["left"])
                top = int(region["top"])
                right = int(region["right"])
                bottom = int(region["bottom"])
            except (KeyError, TypeError, ValueError):
                continue

            left = max(0, min(left, max(0, image.width - 1)))
            top = max(0, min(top, max(0, image.height - 1)))
            right = max(left + 1, min(right, image.width))
            bottom = max(top + 1, min(bottom, image.height))
            prompt_indices.append(prompt_index)
            prompt_boxes.append([left, top, right, bottom])

        if not prompt_boxes:
            continue

        try:
            results = sam_model.predict(
                source=image,
                bboxes=prompt_boxes,
                device=predict_device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("SAM refinement failed on %s: %s", image_path.name, exc)
            continue

        result = results[0] if results else None
        masks = getattr(result, "masks", None) if result is not None else None
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if not isinstance(polygons, (list, tuple)):
            continue

        for local_index, prompt_index in enumerate(prompt_indices):
            if local_index >= len(polygons):
                break
            boundary = polygon_to_boundary_points(
                polygons[local_index],
                image_width=image.width,
                image_height=image.height,
            )
            refined_region = boundary_region_from_points(
                boundary,
                image_width=image.width,
                image_height=image.height,
            )
            if refined_region is None or boundary is None:
                continue

            existing_region = refined_regions[image_index][prompt_index]
            extras = dict(existing_region) if isinstance(existing_region, dict) else {}
            extras.pop("left", None)
            extras.pop("top", None)
            extras.pop("right", None)
            extras.pop("bottom", None)
            extras.pop("boundary", None)
            extras["sam_refined"] = True
            extras["boundary_source"] = "sam"
            refined_regions[image_index][prompt_index] = build_region_payload(
                int(refined_region["left"]),
                int(refined_region["top"]),
                int(refined_region["right"]),
                int(refined_region["bottom"]),
                boundary=boundary,
                extras=extras,
            )

    return refined_regions


def merge_flag_backend_outputs(
    image_paths: list[Path],
    prompt_count: int,
    backend_outputs: list[tuple[np.ndarray, list[list[dict | None]] | None, list[Path]]],
) -> tuple[np.ndarray, list[list[dict | None]]]:
    path_to_index = {path: index for index, path in enumerate(image_paths)}
    merged_scores = np.zeros((len(image_paths), prompt_count), dtype=np.float32)
    merged_regions: list[list[dict | None]] = [[None for _ in range(prompt_count)] for _ in image_paths]

    for scores, regions, paths in backend_outputs:
        if scores.size == 0 or not paths:
            continue
        for source_index, image_path in enumerate(paths):
            target_index = path_to_index.get(image_path)
            if target_index is None:
                continue
            row_scores = np.clip(scores[source_index], 0.0, 1.0).astype(np.float32, copy=False)
            for prompt_index in range(min(prompt_count, row_scores.shape[0])):
                score = float(row_scores[prompt_index])
                if score <= 0.0:
                    continue
                existing_score = float(merged_scores[target_index, prompt_index])
                region = None
                if regions is not None and source_index < len(regions) and prompt_index < len(regions[source_index]):
                    region = regions[source_index][prompt_index]
                if score > existing_score:
                    merged_scores[target_index, prompt_index] = score
                    if region is not None:
                        merged_regions[target_index][prompt_index] = dict(region)
                elif score == existing_score and region is not None and merged_regions[target_index][prompt_index] is None:
                    merged_regions[target_index][prompt_index] = dict(region)

    return merged_scores, merged_regions


def box_to_boundary_points(left: int, top: int, right: int, bottom: int) -> list[dict[str, int]]:
    max_right = max(left, right - 1)
    max_bottom = max(top, bottom - 1)
    return [
        {"x": int(left), "y": int(top)},
        {"x": int(max_right), "y": int(top)},
        {"x": int(max_right), "y": int(max_bottom)},
        {"x": int(left), "y": int(max_bottom)},
    ]


def mask_to_boundary_points(mask: np.ndarray) -> list[dict[str, int]] | None:
    mask_uint8 = np.ascontiguousarray(mask.astype(np.uint8))
    if mask_uint8.ndim != 2 or not mask_uint8.any():
        return None

    contours, _ = cv2.findContours(mask_uint8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return None

    epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    if simplified.shape[0] < 3:
        simplified = contour

    height, width = mask_uint8.shape
    boundary: list[dict[str, int]] = []
    seen_points: set[tuple[int, int]] = set()
    for point in simplified.reshape(-1, 2):
        x = int(np.clip(int(round(float(point[0]))), 0, max(0, width - 1)))
        y = int(np.clip(int(round(float(point[1]))), 0, max(0, height - 1)))
        key = (x, y)
        if key in seen_points:
            continue
        seen_points.add(key)
        boundary.append({"x": x, "y": y})

    return boundary if len(boundary) >= 3 else None


def build_region_payload(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    boundary: list[dict[str, int]] | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    region: dict[str, object] = {
        "left": int(left),
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
    }
    if boundary:
        region["boundary"] = boundary
    if extras:
        region.update(extras)
    return region


def coerce_boundary_points(region: dict, image_width: int, image_height: int) -> list[tuple[int, int]]:
    raw_boundary = region.get("boundary")
    if not isinstance(raw_boundary, list):
        return []

    points: list[tuple[int, int]] = []
    seen_points: set[tuple[int, int]] = set()
    for point in raw_boundary:
        if not isinstance(point, dict):
            continue
        try:
            x = int(round(float(point.get("x", 0))))
            y = int(round(float(point.get("y", 0))))
        except (TypeError, ValueError):
            continue
        x = max(0, min(x, max(0, image_width - 1)))
        y = max(0, min(y, max(0, image_height - 1)))
        key = (x, y)
        if key in seen_points:
            continue
        seen_points.add(key)
        points.append(key)

    return points if len(points) >= 2 else []


def flag_candidate_priority(label: str, region: dict[str, object] | None) -> tuple[int, float]:
    normalized_label = normalize_detector_label(label)
    source = normalize_detector_label(region.get("source", "")) if isinstance(region, dict) else ""
    if normalized_label not in SCENE_FLAG_LABELS and source in LOCALIZED_FLAG_SOURCES:
        return (0, 1.0)
    if normalized_label not in SCENE_FLAG_LABELS:
        return (1, 0.0)
    return (2, 0.0)


def build_image_flag_payload(
    image_paths: list[Path],
    prompt_scores: np.ndarray | None,
    prompt_regions: list[list[dict | None]] | None,
    prompt_texts: list[str],
    prompt_set: str,
    top_k: int,
    min_score: float,
    include_labels: set[str] | None = None,
) -> dict:
    top_k = max(1, int(top_k))
    min_score = float(np.clip(min_score, 0.0, 1.0))
    images_payload: list[dict] = []

    if prompt_scores is None or prompt_scores.size == 0 or len(prompt_scores) != len(image_paths):
        return {
            "prompt_set": prompt_set,
            "top_k": top_k,
            "min_score_percent": similarity_to_percent(min_score),
            "images": images_payload,
        }

    prompt_regions = prompt_regions if prompt_regions is not None and len(prompt_regions) == len(image_paths) else None
    normalized_include_labels = (
        {prompt_text_to_label(label).lower() for label in include_labels if prompt_text_to_label(label)}
        if include_labels
        else None
    )

    for image_index, image_path in enumerate(image_paths):
        image_scores = np.clip(prompt_scores[image_index], 0.0, 1.0).astype(np.float32, copy=False)
        ranked_indices = np.argsort(image_scores)[::-1]
        flags: list[dict] = []
        seen_labels: set[str] = set()
        candidates: list[tuple[tuple[int, float], float, int, str, str, dict | None]] = []

        for prompt_index in ranked_indices:
            score = float(image_scores[prompt_index])
            if score < min_score:
                break

            prompt_text = prompt_texts[int(prompt_index)]
            label = prompt_text_to_label(prompt_text) or prompt_text
            normalized_label = label.lower()
            if normalized_include_labels is not None and normalized_label not in normalized_include_labels:
                continue

            region = prompt_regions[image_index][int(prompt_index)] if prompt_regions is not None else None
            candidates.append(
                (
                    flag_candidate_priority(label, region),
                    -score,
                    int(prompt_index),
                    label,
                    prompt_text,
                    dict(region) if region is not None else None,
                )
            )

        candidates.sort(key=lambda item: (item[0][0], -item[0][1], item[1], item[2]))

        for _, neg_score, _, label, prompt_text, region in candidates:
            if label in seen_labels:
                continue

            flag_entry = {
                "label": label,
                "prompt": prompt_text,
                "score_percent": similarity_to_percent(-neg_score),
            }
            if region is not None:
                flag_entry["region"] = dict(region)
            flags.append(flag_entry)
            seen_labels.add(label)
            if len(flags) >= top_k:
                break

        if not flags and ranked_indices.size and normalized_include_labels is None and float(np.max(image_scores)) >= min_score:
            top_prompt_index = int(ranked_indices[0])
            prompt_text = prompt_texts[top_prompt_index]
            fallback_entry = {
                "label": prompt_text_to_label(prompt_text) or prompt_text,
                "prompt": prompt_text,
                "score_percent": similarity_to_percent(float(image_scores[top_prompt_index])),
            }
            if prompt_regions is not None:
                region = prompt_regions[image_index][top_prompt_index]
                if region is not None:
                    fallback_entry["region"] = dict(region)
            flags.append(fallback_entry)

        images_payload.append({"image": image_path.name, "flagged_items": flags})

    return {
        "prompt_set": prompt_set,
        "top_k": top_k,
        "min_score_percent": similarity_to_percent(min_score),
        "images": images_payload,
    }


def write_image_flags(output_dir: Path, payload: dict) -> dict:
    (output_dir / "image_flags.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def align_flag_backend_output_to_paths(
    image_paths: list[Path],
    scores: np.ndarray,
    regions: list[list[dict | None]] | None,
    resolved_paths: list[Path],
) -> tuple[np.ndarray, list[list[dict | None]]]:
    prompt_count = int(scores.shape[1]) if scores.ndim == 2 else 0
    aligned_scores = np.zeros((len(image_paths), prompt_count), dtype=np.float32)
    aligned_regions: list[list[dict | None]] = [[None for _ in range(prompt_count)] for _ in image_paths]
    path_to_index = {path: index for index, path in enumerate(image_paths)}

    for source_index, image_path in enumerate(resolved_paths):
        target_index = path_to_index.get(image_path)
        if target_index is None:
            continue
        if scores.ndim == 2 and source_index < len(scores):
            aligned_scores[target_index] = np.clip(scores[source_index], 0.0, 1.0).astype(np.float32, copy=False)
        if regions is not None and source_index < len(regions):
            aligned_regions[target_index] = [
                dict(region) if region is not None else None
                for region in regions[source_index][:prompt_count]
            ] + [None for _ in range(max(0, prompt_count - len(regions[source_index])))]

    return aligned_scores, aligned_regions


def extract_item_flag_outputs(
    image_paths: list[Path],
    args: argparse.Namespace,
    *,
    device: str,
    cache_dir: Path,
    prompt_texts: list[str],
    feature_cache: dict[Path, CachedImageFeatures] | None,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[list[dict | None]]]:
    if not image_paths or not args.flag_items:
        return np.empty((0, 0), dtype=np.float32), []

    if args.flag_detector == "yolo_scene_clip":
        scores, regions, resolved_paths = extract_yolo_scene_clip_flag_scores(
            image_paths=image_paths,
            model_name=args.model,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            yolo_model_path=str(args.yolo_model),
            yolo_confidence_threshold=float(args.yolo_confidence),
            yolo_iou_threshold=float(args.yolo_iou),
            yolo_image_size=int(args.yolo_imgsz),
            yolo_max_detections=int(args.yolo_max_det),
            yolo_retina_masks=bool(args.yolo_retina_masks),
            scene_clip_min_score=float(args.scene_clip_min_score),
        )
    elif args.flag_detector == "hybrid":
        scores, regions, resolved_paths = extract_hybrid_flag_scores(
            image_paths=image_paths,
            model_name=args.model,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            yolo_model_path=str(args.yolo_model),
            yolo_confidence_threshold=float(args.yolo_confidence),
            yolo_iou_threshold=float(args.yolo_iou),
            yolo_image_size=int(args.yolo_imgsz),
            yolo_max_detections=int(args.yolo_max_det),
            yolo_retina_masks=bool(args.yolo_retina_masks),
            open_vocab_model_id=str(args.open_vocab_model),
            open_vocab_score_threshold=float(args.open_vocab_threshold),
            segmentation_min_area_ratio=float(args.segmentation_min_area),
        )
    elif args.flag_detector == "sam_deeplab_yolo_clip":
        scores, regions, resolved_paths = extract_sam_deeplab_yolo_clip_flag_scores(
            image_paths=image_paths,
            model_name=args.model,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            yolo_model_path=str(args.yolo_model),
            yolo_confidence_threshold=float(args.yolo_confidence),
            yolo_iou_threshold=float(args.yolo_iou),
            yolo_image_size=int(args.yolo_imgsz),
            yolo_max_detections=int(args.yolo_max_det),
            yolo_retina_masks=bool(args.yolo_retina_masks),
            deeplab_model_name=str(args.deeplab_model),
            deeplab_min_area_ratio=float(args.deeplab_min_area),
            sam_model_path=str(args.sam_model),
        )
    elif args.flag_detector == "open_vocab_hybrid":
        scores, regions, resolved_paths = extract_open_vocab_hybrid_flag_scores(
            image_paths=image_paths,
            model_name=args.model,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            open_vocab_model_id=str(args.open_vocab_model),
            open_vocab_score_threshold=float(args.open_vocab_threshold),
            segmentation_min_area_ratio=float(args.segmentation_min_area),
        )
    elif args.flag_detector == "open_vocab":
        scores, regions, resolved_paths = extract_open_vocab_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_id=str(args.open_vocab_model),
            score_threshold=float(args.open_vocab_threshold),
        )
    elif args.flag_detector == "yolo":
        scores, regions, resolved_paths = extract_yolo_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            model_path=str(args.yolo_model),
            confidence_threshold=float(args.yolo_confidence),
            iou_threshold=float(args.yolo_iou),
            image_size=int(args.yolo_imgsz),
            max_detections=int(args.yolo_max_det),
            retina_masks=bool(args.yolo_retina_masks),
        )
    elif args.flag_detector == "segformer_ade20k":
        scores, regions, resolved_paths = extract_segmentation_flag_scores(
            image_paths=image_paths,
            device=device,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
            min_area_ratio=float(args.segmentation_min_area),
        )
    else:
        scores, regions, resolved_paths = extract_clip_flag_scores(
            image_paths=image_paths,
            model_name=args.model,
            device=device,
            cache_dir=cache_dir,
            prompt_texts=prompt_texts,
            feature_cache=feature_cache,
            logger=logger,
        )

    return align_flag_backend_output_to_paths(image_paths, scores, regions, resolved_paths)


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
        "ultralytics_available": bool(importlib.util.find_spec("ultralytics")),
        "transformers_available": bool(importlib.util.find_spec("transformers")),
        "open_vocab_model": DEFAULT_OPEN_VOCAB_MODEL,
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
    image_output_paths = {str(key): str(value) for key, value in result.get("image_output_paths", {}).items()}
    show_item_flags = bool(manifest.get("settings", {}).get("flag_items"))

    def render_flagged_items_html(flagged_items: list[dict] | None) -> str:
        if not flagged_items:
            return '<span class="muted">No item flags</span>'
        return "".join(
            f'<span class="chip">{escape(str(item["label"]))} ({escape(str(item["score_percent"]))}%)</span>'
            for item in flagged_items
        )

    def render_image_name_html(image_name: str | None) -> str:
        if image_name is None:
            return "None"
        label = escape(str(image_name))
        relative_path = image_output_paths.get(str(image_name))
        if not relative_path:
            return label
        return f'<a href="{escape(relative_path)}" target="_blank" rel="noopener noreferrer">{label}</a>'

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
            f"<li>{render_image_name_html(item['image'])} ({item['match_percent']}%)</li>" for item in near_misses
        ) or "<li>None</li>"
        representative_flags_html = ""
        if show_item_flags:
            representative_flags_html = render_flagged_items_html(
                image_matches.get(representative, {}).get("flagged_items") if representative is not None else None
            )
            images_html = "".join(
                (
                    f"<li><strong>{render_image_name_html(image_name)}</strong>"
                    f"<div class=\"chips\">{render_flagged_items_html(image_matches.get(image_name, {}).get('flagged_items'))}</div></li>"
                )
                for image_name in images
            )
        else:
            images_html = "".join(f"<li>{render_image_name_html(image_name)}</li>" for image_name in images)
        representative_flags_block = (
            f"<p><strong>Representative Flags:</strong> <span class=\"chips\">{representative_flags_html}</span></p>"
            if show_item_flags
            else ""
        )
        cluster_sections.append(
            """
            <section class="cluster-card">
              <h2>Cluster {cluster_id}</h2>
              {contact_html}
              <p><strong>Representative:</strong> {representative}</p>
              {representative_flags_block}
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
                representative=render_image_name_html(representative),
                representative_flags_block=representative_flags_block,
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
        flags_fragment = ""
        if show_item_flags:
            flagged_items_html = render_flagged_items_html(image_matches.get(noise_item["image"], {}).get("flagged_items"))
            flags_fragment = f"Flags: <span class=\"chips\">{flagged_items_html}</span>. "
        noise_cards.append(
            f"<li><strong>{render_image_name_html(noise_item['image'])}</strong>: {escape(reasons)}. "
            f"{flags_fragment}"
            f"Best candidate: {escape(candidate_text)}</li>"
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
    a {{ color: #0f4c5c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    ul {{ margin-top: 8px; padding-left: 20px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }}
    .chip {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #e2ecf5; color: #16324f; font-size: 0.9rem; }}
    .muted {{ color: #6b7280; }}
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
    image_flag_lookup: dict[str, list[dict]] | None = None,
    annotate_flagged_images: bool = False,
) -> dict:
    clusters_payload: list[dict] = []
    noise_images: list[str] = []
    noise_details_payload: list[dict] = []
    noise_contact_sheet: str | None = None
    image_output_paths: dict[str, str] = {}
    image_flag_lookup = image_flag_lookup or {}

    unique_labels = sorted(set(int(label) for label in labels))
    for label in unique_labels:
        members = [path for path, cluster_label in zip(image_paths, labels) if int(cluster_label) == label]
        if not members:
            continue

        if label == -1:
            noise_dir = output_dir / "noise"
            noise_dir.mkdir(parents=True, exist_ok=True)
            exported_noise_paths: list[Path] = []
            for image_path in members:
                destination_path = noise_dir / image_path.name
                export_cluster_image(
                    image_path=image_path,
                    destination_path=destination_path,
                    flagged_items=image_flag_lookup.get(image_path.name),
                    annotate_flagged_images=annotate_flagged_images,
                    feature_cache=feature_cache,
                )
                noise_images.append(image_path.name)
                exported_noise_paths.append(destination_path)
                image_output_paths[image_path.name] = destination_path.relative_to(output_dir).as_posix()
                if noise_details is not None and image_path.name in noise_details:
                    noise_details_payload.append(
                        {
                            "image": image_path.name,
                            **noise_details[image_path.name],
                        }
                    )
            if generate_contact_sheets:
                contact_sheet_path = save_contact_sheet(
                    exported_noise_paths if annotate_flagged_images else members,
                    output_dir / "noise_contact.jpg",
                    feature_cache=None if annotate_flagged_images else feature_cache,
                )
                if contact_sheet_path is not None:
                    noise_contact_sheet = contact_sheet_path.name
            continue

        cluster_dir = output_dir / f"cluster_{label}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        exported_member_paths: list[Path] = []
        for image_path in members:
            destination_path = cluster_dir / image_path.name
            export_cluster_image(
                image_path=image_path,
                destination_path=destination_path,
                flagged_items=image_flag_lookup.get(image_path.name),
                annotate_flagged_images=annotate_flagged_images,
                feature_cache=feature_cache,
            )
            exported_member_paths.append(destination_path)
            image_output_paths[image_path.name] = destination_path.relative_to(output_dir).as_posix()

        contact_sheet_name: str | None = None
        if generate_contact_sheets:
            contact_sheet_path = save_contact_sheet(
                exported_member_paths if annotate_flagged_images else members,
                output_dir / f"cluster_{label}_contact.jpg",
                feature_cache=None if annotate_flagged_images else feature_cache,
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
        "image_flags": image_flag_lookup,
        "image_output_paths": image_output_paths,
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
    image_flag_lookup: dict[str, list[dict]] | None,
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
        image_flag_lookup=image_flag_lookup,
        logger=logger,
    )
    (output_dir / "match_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    args = apply_preset_defaults(args, sys.argv[1:])
    logger = setup_logging()
    annotate_flagged_images = resolve_annotate_flagged_images(args)

    if annotate_flagged_images and not args.flag_items:
        raise SystemExit("--annotate-flagged-images requires --flag-items.")

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
    resolved_settings = build_resolved_run_settings(
        args,
        input_dir,
        requested_output_dir,
        output_dir,
        device,
        annotate_flagged_images,
    )

    logger.info("Reading images from %s", input_dir)
    image_paths = discover_images(input_dir)
    if not image_paths:
        raise SystemExit("No supported images found in input folder.")

    logger.info("Found %s images", len(image_paths))
    feature_cache, cached_paths, skipped_images = build_feature_cache(image_paths, logger)
    if skipped_images:
        logger.info("Skipped %s unreadable images during feature cache build", len(skipped_images))
    prompt_texts = get_prompt_texts(args.prompt_set)
    flag_prompt_texts = get_prompt_texts(args.flag_prompt_set)
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
    item_prompt_scores: np.ndarray | None = None
    item_prompt_regions: list[list[dict | None]] | None = None
    direct_image_flags_payload: dict | None = None
    if args.strict_same_corner_items:
        logger.info("Extracting CLIP prompt signatures for strict same-corner+items mode")
        item_features, _, item_paths = extract_clip_item_features(
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
        if item_prompt_scores is not None:
            item_prompt_scores = item_prompt_scores[embedded_indices]
        if item_prompt_regions is not None:
            item_prompt_regions = [item_prompt_regions[index] for index in embedded_indices]

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

    if args.flag_items:
        detector_label = (
            "YOLOv8n-seg + heuristic scene CLIP"
            if args.flag_detector == "yolo_scene_clip"
            else "adaptive hybrid YOLOv8 + OWLv2 + SegFormer + CLIP"
            if args.flag_detector == "hybrid"
            else "SAM + DeepLabV3 + YOLOv8 + CLIP"
            if args.flag_detector == "sam_deeplab_yolo_clip"
            else "OWLv2 + SegFormer + CLIP"
            if args.flag_detector == "open_vocab_hybrid"
            else "OWLv2 open-vocabulary detection"
            if args.flag_detector == "open_vocab"
            else "YOLOv8 segmentation"
            if args.flag_detector == "yolo"
            else "crop-aware CLIP"
            if args.flag_detector == "clip"
            else "semantic segmentation"
        )
        logger.info(
            "Extracting %s item flags using prompt set %s after clustering",
            detector_label,
            args.flag_prompt_set,
        )
        if args.flag_detector == "yolo_scene_clip":
            direct_image_flags_payload = build_yolo_scene_clip_image_flag_payload(
                image_paths=embedded_paths,
                prompt_texts=flag_prompt_texts,
                prompt_set=args.flag_prompt_set,
                model_name=args.model,
                device=device,
                cache_dir=cache_dir,
                feature_cache=feature_cache,
                logger=logger,
                yolo_model_path=str(args.yolo_model),
                yolo_confidence_threshold=float(args.yolo_confidence),
                yolo_iou_threshold=float(args.yolo_iou),
                yolo_image_size=int(args.yolo_imgsz),
                yolo_max_detections=int(args.yolo_max_det),
                yolo_retina_masks=bool(args.yolo_retina_masks),
                top_k=args.flag_top_k,
                min_score=float(args.flag_min_score),
                scene_clip_min_score=float(args.scene_clip_min_score),
                include_labels=set(args.flag_include_labels or []),
            )
        else:
            item_prompt_scores, item_prompt_regions = extract_item_flag_outputs(
                image_paths=embedded_paths,
                args=args,
                device=device,
                cache_dir=cache_dir,
                prompt_texts=flag_prompt_texts,
                feature_cache=feature_cache,
                logger=logger,
            )

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    image_flags_payload = (
        direct_image_flags_payload
        if args.flag_items and direct_image_flags_payload is not None
        else build_image_flag_payload(
            image_paths=embedded_paths,
            prompt_scores=item_prompt_scores,
            prompt_regions=item_prompt_regions,
            prompt_texts=flag_prompt_texts,
            prompt_set=args.flag_prompt_set,
            top_k=args.flag_top_k,
            min_score=args.flag_min_score,
            include_labels=set(args.flag_include_labels or []),
        )
        if args.flag_items
        else {
            "prompt_set": args.flag_prompt_set,
            "top_k": max(1, int(args.flag_top_k)),
            "min_score_percent": similarity_to_percent(float(args.flag_min_score)),
            "images": [],
        }
    )
    if args.flag_items:
        write_image_flags(output_dir, image_flags_payload)
    image_flag_lookup = {
        entry["image"]: entry.get("flagged_items", []) for entry in image_flags_payload.get("images", [])
    }
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
        image_flag_lookup=image_flag_lookup,
        annotate_flagged_images=annotate_flagged_images,
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
        image_flag_lookup=image_flag_lookup,
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
