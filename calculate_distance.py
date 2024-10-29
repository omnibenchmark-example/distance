import argparse
import os
from pathlib import Path

import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances


VALID_MEASURES = ["cosine", "euclidean", "manhattan", "chebyshev"]


def calculate_distance_matrix(data, measure):
    if measure == "cosine":
        distance_matrix = 1 - cosine_similarity(data)
    elif measure == "euclidean":
        distance_matrix = euclidean_distances(data)
    elif measure == "manhattan":
        distance_matrix = manhattan_distances(data)
    elif measure == "chebyshev":
        distance_matrix = cdist(data, data, metric='chebyshev')
    else:
        raise ValueError("Invalid measure")

    return distance_matrix


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Calculate pairwise distance matrix of features.')

    # Add arguments
    parser.add_argument('--data.features', type=str, help='features for which the distance calculation takes place.', required=True)
    parser.add_argument('--measure', type=str, help='distance measure to apply. Choose from: `cosine`, `euclidean`, `manhattan`, `chebyshev`.', default='euclidean')
    parser.add_argument('--output_dir', type=str, help='output directory where dataset files will be saved.', default=os.getcwd())
    parser.add_argument('--name', type=str, help='name of the dataset', default='distance')

    # Parse arguments
    args = parser.parse_args()

    if args.measure not in VALID_MEASURES:
        raise ValueError(f"Invalid measure `{args.measure}`. Choose from: `cosine`, `euclidean`, `manhattan`, `chebyshev`.")

    features_df = pd.read_csv(getattr(args, 'data.features'))
    data = features_df.loc[:, features_df.columns != 'id'].values
    distances = calculate_distance_matrix(data, args.measure)
    distance_df = pd.DataFrame(distances, index=features_df['id'], columns=features_df['id'])

    # Write distances to disk
    distance_df.to_csv(Path(args.output_dir) / f'{args.name}.distances.csv', index_label='id')


if __name__ == "__main__":
    main()
