import os
import cv2
import numpy as np
import shutil
import argparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def load_and_preprocess_images(input_dir, resize_dim=(256, 256)):
    """
    Loads images from the input directory, converts them to grayscale,
    applies histogram equalization, resizes them, and flattens them into vectors.
    """
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []

    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            image_paths.append(os.path.join(input_dir, fname))

    if not image_paths:
        print(f"No valid images found in {input_dir}")
        return [], []

    print(f"Found {len(image_paths)} images. Processing...\n")

    vectors = []
    valid_paths = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image {path}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to normalize lighting differences
        equalized = cv2.equalizeHist(gray)

        # Resize to uniform dimensions
        resized = cv2.resize(equalized, resize_dim)

        # Flatten into 1D vector
        flattened = resized.flatten().astype(np.float32)

        vectors.append(flattened)
        valid_paths.append(path)

    return valid_paths, np.array(vectors)


def print_similarity_matrix(image_paths, vectors):
    """
    Prints the pairwise cosine similarity matrix so the user can
    see how similar each pair of images is and tune the threshold.
    """
    sim_matrix = cosine_similarity(vectors)
    names = [os.path.basename(p) for p in image_paths]

    # Print header
    max_name_len = max(len(n) for n in names)
    header = " " * (max_name_len + 2)
    for i, name in enumerate(names):
        header += f" {i:>5}"
    print("=== Cosine Similarity Matrix ===")
    print(f"(Index mapping below)\n")

    # Print matrix with indices
    for i in range(len(names)):
        row = f"  [{i:>2}] "
        for j in range(len(names)):
            row += f" {sim_matrix[i][j]:.3f}"
        print(row)

    print(f"\n  Index -> Image mapping:")
    for i, name in enumerate(names):
        print(f"  [{i:>2}] {name}")
    print()

    return sim_matrix


def cluster_and_find_best(image_paths, vectors, threshold, output_dir):
    """
    Clusters the images using cosine similarity and finds the 'best' representative
    image for each cluster. Outputs only the best images into a single folder.
    """
    if len(vectors) == 0:
        print("No image vectors to cluster.")
        return

    # Print similarity matrix for user reference
    sim_matrix = print_similarity_matrix(image_paths, vectors)

    # Cosine distance = 1 - cosine_similarity
    distance_threshold = 1.0 - threshold

    print(f"Clustering {len(vectors)} images with similarity threshold {threshold} "
          f"(distance threshold {distance_threshold:.4f})...\n")

    # Use 'average' linkage with 'cosine' distance metric
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )

    labels = clustering.fit_predict(vectors)
    unique_clusters = np.unique(labels)

    print(f"Found {len(unique_clusters)} distinct clusters.\n")

    # Clean and create output directory for best images
    best_images_dir = os.path.join(output_dir, "best_images")
    if os.path.exists(best_images_dir):
        shutil.rmtree(best_images_dir)
    os.makedirs(best_images_dir, exist_ok=True)

    # For each cluster, find the best image
    for cluster_id in unique_clusters:
        # Get indices of images in this cluster
        indices = np.where(labels == cluster_id)[0]

        # Get the vectors for this cluster
        cluster_vectors = vectors[indices]

        # Compute the centroid of the cluster
        centroid = np.mean(cluster_vectors, axis=0).reshape(1, -1)

        # Find the image closest to the centroid (using cosine distance)
        distances_to_centroid = cosine_distances(cluster_vectors, centroid).flatten()
        best_idx_local = np.argmin(distances_to_centroid)
        best_idx_global = indices[best_idx_local]
        best_image_path = image_paths[best_idx_global]

        # Print cluster details
        cluster_image_names = [os.path.basename(image_paths[i]) for i in indices]
        print(f"  Cluster {cluster_id} ({len(indices)} images): {cluster_image_names}")
        print(f"    -> Best image: {os.path.basename(best_image_path)}\n")

        # Copy the best image into the best_images output directory
        best_dest = os.path.join(best_images_dir, os.path.basename(best_image_path))
        shutil.copy2(best_image_path, best_dest)

    print(f"Output: {len(unique_clusters)} best images saved to {best_images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster home images by visual similarity and output the best representative image per cluster."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("-o", "--output", type=str, default="./output",
                        help="Directory to save best images output")
    parser.add_argument("-t", "--threshold", type=float, default=0.95,
                        help="Cosine similarity threshold (default: 0.95). "
                             "Higher = stricter = more clusters.")
    parser.add_argument("--size", type=int, default=256,
                        help="Resize dimension before vectorizing (default: 256 for 256x256).")

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    threshold = args.threshold
    resize_dim = (args.size, args.size)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    image_paths, vectors = load_and_preprocess_images(input_dir, resize_dim)

    if len(image_paths) > 0:
        cluster_and_find_best(image_paths, vectors, threshold, output_dir)


if __name__ == "__main__":
    main()
