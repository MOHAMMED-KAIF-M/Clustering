import os
import json
import shutil
import cv2
import numpy as np
import argparse
import csv
from sklearn.cluster import AgglomerativeClustering
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# ---------------------------------------------------------------------------
# Feature Extraction: ORB (Geometric) + CLIP (Semantic), both on Grayscale
# ---------------------------------------------------------------------------

class ORBFeatureMatcher:
    """Extracts ORB keypoints from grayscale images and computes match scores."""
    def __init__(self, nfeatures=2500):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract(self, gray_img):
        kp, des = self.orb.detectAndCompute(gray_img, None)
        return kp, des

    def get_match_score(self, des1, des2):
        if des1 is None or len(des1) < 2 or des2 is None or len(des2) < 2:
            return 0.0
        matches_12 = self.bf.knnMatch(des1, des2, k=2)
        good_12 = sum(1 for p in matches_12 if len(p) == 2 and p[0].distance < 0.80 * p[1].distance)
        matches_21 = self.bf.knnMatch(des2, des1, k=2)
        good_21 = sum(1 for p in matches_21 if len(p) == 2 and p[0].distance < 0.80 * p[1].distance)
        min_kpts = min(len(des1), len(des2))
        return min(max(good_12, good_21) / float(min_kpts), 1.0) if min_kpts > 0 else 0.0


class CLIPFeatureExtractor:
    """Extracts semantic embeddings from grayscale images using CLIP ViT."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def extract(self, gray_img):
        # Create fake RGB from grayscale (forces CLIP to focus on shapes, not colors)
        fake_rgb = cv2.merge([gray_img, gray_img, gray_img])
        pil_img = Image.fromarray(fake_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs["pixel_values"].unsqueeze(0) if inputs["pixel_values"].dim() == 3 else inputs)
            # Use the pooler_output (CLS token embedding)
            image_features = outputs.pooler_output
        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_images(input_dir):
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            image_paths.append(os.path.join(input_dir, fname))
    return image_paths


def prepare_grayscale(img_bgr, max_dim=800):
    """Resize and convert to grayscale."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def extract_all_features(image_paths):
    """Extract both ORB descriptors and CLIP embeddings for every image."""
    print(f"Found {len(image_paths)} images.\n")

    orb_matcher = ORBFeatureMatcher()
    clip_extractor = CLIPFeatureExtractor()

    orb_descriptors = []
    clip_embeddings = []
    valid_paths = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  Warning: Could not read {path}")
            continue

        gray = prepare_grayscale(img)

        # ORB features
        kp, des = orb_matcher.extract(gray)
        orb_descriptors.append(des)

        # CLIP features
        embedding = clip_extractor.extract(gray)
        clip_embeddings.append(embedding)

        valid_paths.append(path)
        name = os.path.basename(path)
        print(f"  {name}: ORB={len(kp) if kp else 0} keypoints, CLIP=512-dim vector")

    print()
    return valid_paths, orb_descriptors, clip_embeddings


def compute_combined_distance_matrix(image_paths, orb_descriptors, clip_embeddings, orb_weight=0.4, clip_weight=0.6):
    """
    Compute a combined similarity matrix:
      combined_sim = orb_weight * orb_sim + clip_weight * clip_sim
    ORB captures local geometric structure; CLIP captures global semantic content.
    """
    n = len(image_paths)
    orb_matcher = ORBFeatureMatcher()

    orb_sim = np.zeros((n, n), dtype=np.float32)
    clip_sim = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                orb_sim[i, j] = 1.0
                clip_sim[i, j] = 1.0
            else:
                # ORB similarity
                o_score = orb_matcher.get_match_score(orb_descriptors[i], orb_descriptors[j])
                orb_sim[i, j] = o_score
                orb_sim[j, i] = o_score

                # CLIP cosine similarity
                c_score = float(np.dot(clip_embeddings[i], clip_embeddings[j]))
                c_score = max(min(c_score, 1.0), 0.0)
                clip_sim[i, j] = c_score
                clip_sim[j, i] = c_score

    # Absolute Normalization for stable thresholds across entirely different datasets
    # ORB typically saturates around 0.10 for an amazing geometric match
    orb_sim_norm = np.clip(orb_sim / 0.10, 0.0, 1.0)
    np.fill_diagonal(orb_sim_norm, 1.0)

    # CLIP typically ranges 0.70 to 0.95 (below 0.7 is basically random objects)
    clip_sim_norm = np.clip((clip_sim - 0.70) / (0.95 - 0.70), 0.0, 1.0)
    np.fill_diagonal(clip_sim_norm, 1.0)

    # Weighted combination
    combined_sim = orb_weight * orb_sim_norm + clip_weight * clip_sim_norm
    np.fill_diagonal(combined_sim, 1.0)

    # Convert to distance
    dist_matrix = 1.0 - combined_sim
    np.fill_diagonal(dist_matrix, 0.0)

    names = [os.path.basename(p) for p in image_paths]

    print("=== ORB Geometric Similarity (raw) ===\n")
    for i in range(n):
        row = f"  [{i:>2}] "
        for j in range(n):
            row += f" {orb_sim[i][j]:.3f}"
        print(row)

    print("\n=== CLIP Semantic Similarity (raw) ===\n")
    for i in range(n):
        row = f"  [{i:>2}] "
        for j in range(n):
            row += f" {clip_sim[i][j]:.3f}"
        print(row)

    print("\n=== Combined Similarity (ORB {:.0f}% + CLIP {:.0f}%) ===\n".format(orb_weight * 100, clip_weight * 100))
    for i in range(n):
        row = f"  [{i:>2}] "
        for j in range(n):
            row += f" {combined_sim[i][j]:.3f}"
        print(row)

    print(f"\n  Index -> Image mapping:")
    for i, name in enumerate(names):
        print(f"  [{i:>2}] {name}")
    print()

    return dist_matrix, combined_sim

def save_match_scores_csv(image_paths, combined_sim, output_csv):
    """Saves the combined similarity matrix to a CSV file."""
    names = [os.path.basename(p) for p in image_paths]
    
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow([''] + names)
        
        # Write rows
        for i in range(len(names)):
            row = [names[i]] + [f"{score:.4f}" for score in combined_sim[i]]
            writer.writerow(row)
            
    print(f"=== Match score matrix successfully saved to {output_csv} ===\n")


def form_clusters(image_paths, orb_descriptors, clip_embeddings, threshold,
                  orb_weight=0.6, clip_weight=0.4,
                  output_json=None, output_dir=None, output_csv=None):
    if len(orb_descriptors) == 0:
        print("No features to cluster.")
        return

    dist_matrix, combined_sim = compute_combined_distance_matrix(
        image_paths, orb_descriptors, clip_embeddings, orb_weight, clip_weight
    )

    if output_csv:
        save_match_scores_csv(image_paths, combined_sim, output_csv)

    distance_threshold = 1.0 - threshold

    print(f"Clustering with combined similarity threshold: {threshold} "
          f"(distance threshold: {distance_threshold:.4f})\n")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='precomputed',
        linkage='average'
    )

    labels = clustering.fit_predict(dist_matrix)
    unique_clusters = np.unique(labels)

    print(f"{'=' * 50}")
    print(f"  RESULT: {len(unique_clusters)} distinct groups formed")
    print(f"{'=' * 50}\n")

    cluster_report = {}

    for cluster_id in unique_clusters:
        indices = np.where(labels == cluster_id)[0]
        cluster_images = [os.path.basename(image_paths[i]) for i in indices]
        cluster_report[f"cluster_{cluster_id}"] = cluster_images

        print(f"  Cluster {cluster_id} ({len(indices)} images):")

        if output_dir:
            cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_dir, exist_ok=True)

        for i in indices:
            img_name = os.path.basename(image_paths[i])
            print(f"    - {img_name}")
            if output_dir:
                try:
                    shutil.copy2(image_paths[i], os.path.join(cluster_dir, img_name))
                except Exception as e:
                    print(f"    Error copying {img_name}: {e}")
        print()

    if output_json:
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(cluster_report, f, indent=2)
        print(f"Cluster report saved to {output_json}")

    return cluster_report


def main():
    parser = argparse.ArgumentParser(description="Grayscale + ORB + CLIP Ensemble Image Clustering")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory")
    parser.add_argument("-t", "--threshold", type=float, default=0.55,
                        help="Combined similarity threshold. Higher = more clusters.")
    parser.add_argument("--orb-weight", type=float, default=0.6,
                        help="Weight for ORB geometric similarity (0.0-1.0)")
    parser.add_argument("--clip-weight", type=float, default=0.4,
                        help="Weight for CLIP semantic similarity (0.0-1.0)")
    parser.add_argument("-o", "--output-json", type=str, default=None)
    parser.add_argument("-d", "--output-dir", type=str, default=None,
                        help="Directory to save clustered images")
    parser.add_argument("-c", "--output-csv", type=str, default=None,
                        help="Path to save the match score similarity matrix as a CSV file")

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    image_paths = load_images(args.input)
    if not image_paths:
        print("No images found.")
        return

    valid_paths, orb_descs, clip_embeds = extract_all_features(image_paths)

    if len(valid_paths) > 0:
        form_clusters(valid_paths, orb_descs, clip_embeds, args.threshold,
                      args.orb_weight, args.clip_weight,
                      args.output_json, args.output_dir, args.output_csv)


if __name__ == "__main__":
    main()
