# Usage:
#   python scripts/generate_landmark_distances.py --data_dir data/OSM_Beijing --num_landmarks 61
#   python scripts/generate_landmark_distances.py --data_dir data/OSM_Harbin_Small --num_landmarks 61

import os
import sys
import argparse

# Auto-detect project root (scripts/ -> l1tilde-metric-study/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from utils.data_utils import (
    load_graph,
    print_summary_stats,
    print_green,
    print_warning,
    select_landmarks,
    compute_landmark_distances,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate landmark distance embeddings for shortest-distance datasets.")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to directory containing *.nodes and *.edges files '
                             '(e.g., data/OSM_Harbin_Small)')
    parser.add_argument('--num_landmarks', type=int, default=61,
                        help='Number of landmarks to select')
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(PROJECT_DIR, 'data', 'OSM_Harbin_Small')
        print(f"No --data_dir provided, using default: {args.data_dir}")

    if not os.path.exists(args.data_dir):
        print(f"Data directory '{args.data_dir}' does not exist.")
        print(f"Run 'python scripts/prepare_data.py' first to download data.")
        sys.exit(1)

    data_name = os.path.basename(os.path.normpath(args.data_dir))
    print(f"Data: {data_name}")
    print(f"Data dir: {args.data_dir}")
    print(f"Num landmarks: {args.num_landmarks}")

    # Load graph
    G = load_graph(args.data_dir)
    print_summary_stats(G)

    # Select landmarks and compute distances
    landmarks = select_landmarks(G, args.num_landmarks, strategy="random")
    dist_matrix = compute_landmark_distances(G, landmarks)
    print(f"Computed landmark distance matrix with shape: {dist_matrix.shape}")

    # Save landmark distance embeddings
    node_attr_path = os.path.join(args.data_dir, f"landmark_dim{args.num_landmarks}.embeddings")
    print_green(f"Saving nodes: {node_attr_path}")
    print_warning("Warning: The node ids are right-shifted by 1 (i.e., node ids start from `1 to n` "
                  "instead of `0 to n-1`) in the saved files.")
    comment = "#"
    delimiter = " "
    with open(node_attr_path, 'w') as f:
        f.write(f"{comment} Format: node_id features\n")
        for node, data in enumerate(dist_matrix):
            f.write(f"{node+1}{delimiter}{delimiter.join(map(str, data))}\n")  # Right-shift node id by 1

    print_green(f"Done! Embeddings saved to: {node_attr_path}")
