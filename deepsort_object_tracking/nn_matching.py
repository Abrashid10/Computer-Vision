import numpy as np


def cosine_distance(a, b, data_is_normalized=False):
    """
    Compute cosine distance between rows of a and b.

    a: (N, D) matrix
    b: (M, D) matrix

    Returns: (N, M) distance matrix
    """
    if not data_is_normalized:
        # L2 normalize each row
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)

    # Cosine similarity = a @ bᵀ
    # Cosine distance   = 1 - similarity
    return 1.0 - np.dot(a, b.T)


def nn_cosine_distance(x, y):
    """
    Nearest-neighbor cosine distance.

    x: gallery features (N_gallery, D)
    y: query features   (N_query, D)

    For each query, find minimum distance to gallery.
    Returns: (N_gallery,) array of minimum distances
    """
    distances = cosine_distance(x, y)          # (N_gallery, N_query)
    return distances.min(axis=0)               # min over gallery → (N_query,)


class NearestNeighborDistanceMetric:
    """
    Maintains a feature gallery per track.
    Computes nearest-neighbor cosine distance for matching.
    """

    def __init__(self, matching_threshold, budget=None):
        """
        matching_threshold: max cosine distance for valid match
                            paper uses 0.2
        budget:             max features to store per track (None=unlimited)
                            paper uses 100
        """
        self.matching_threshold = matching_threshold
        self.budget             = budget
        self.samples            = {}    # track_id → list of features

    def partial_fit(self, features, targets, active_targets):
        """
        Update gallery with new features.

        features:       (N, 128) new embeddings
        targets:        (N,) track IDs for each feature
        active_targets: list of currently active track IDs
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, [])
            self.samples[target].append(feature)

            # Keep only last 'budget' features per track
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

        # Remove features for inactive tracks (save memory)
        self.samples = {
            k: v for k, v in self.samples.items()
            if k in active_targets
        }

    def distance(self, features, targets):
        """
        Compute cost matrix between detections and tracks.

        features: (N, 128) detection embeddings
        targets:  list of track IDs

        Returns: (len(targets), N) cost matrix
        """
        cost_matrix = np.zeros((len(targets), len(features)))

        for i, target in enumerate(targets):
            gallery = np.array(self.samples[target])   # (K, 128)
            # Min cosine distance over gallery
            cost_matrix[i] = nn_cosine_distance(gallery, features)

        return cost_matrix
