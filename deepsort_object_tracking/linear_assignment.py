import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import chi2inv95

INFTY_COST = 1e5    # represents impossible assignment


def min_cost_matching(distance_metric, max_distance,
                      tracks, detections,
                      track_indices=None,
                      detection_indices=None):
    """
    Solve linear assignment problem.

    Returns: (matches, unmatched_tracks, unmatched_detections)
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices

    # Build cost matrix
    cost_matrix = distance_metric(
        tracks, detections,
        track_indices, detection_indices
    )

    # Set costs above threshold to INFTY (impossible)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches            = []
    unmatched_tracks   = list(track_indices)
    unmatched_dets     = list(detection_indices)

    for r, c in zip(row_ind, col_ind):
        track_idx = track_indices[r]
        det_idx   = detection_indices[c]

        if cost_matrix[r, c] > max_distance:
            # Cost too high → unmatched
            continue

        matches.append((track_idx, det_idx))
        unmatched_tracks.remove(track_idx)
        unmatched_dets.remove(det_idx)

    return matches, unmatched_tracks, unmatched_dets


def matching_cascade(distance_metric, max_distance,
                     cascade_depth, tracks, detections,
                     track_indices=None, detection_indices=None):
    """
    Cascade matching — prioritize recently seen tracks.

    cascade_depth = max_age (70 in paper)

    For age=1,2,...,max_age:
        Try to match tracks of this age to remaining detections
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches = []

    # Priority: tracks missed for fewer frames go first
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break   # all detections matched

        # Tracks missed for exactly 'level+1' frames
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == level + 1
        ]

        if len(track_indices_l) == 0:
            continue   # no tracks at this age level

        # Match these tracks to remaining unmatched detections
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance,
            tracks, detections,
            track_indices_l, unmatched_detections
        )

        matches += matches_l

    # Remaining unmatched tracks
    unmatched_tracks = list(set(track_indices) - {k for k, _ in matches})

    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections,
                     track_indices, detection_indices,
                     gated_cost=INFTY_COST,
                     only_position=False):
    """
    Apply Mahalanobis gating to cost matrix.
    Set impossible matches (too far physically) to INFTY.
    """
    # Threshold from chi-squared distribution
    # 4 degrees of freedom (measurement dim)
    gating_threshold = chi2inv95[4]

    # Get all detection measurements
    measurements = np.array([
        detections[i].to_xyah()
        for i in detection_indices
    ])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]

        # Compute Mahalanobis distances
        gating_distance = kf.gating_distance(
            track.mean, track.covariance,
            measurements, only_position
        )

        # Gate: set cost to INFTY if physically impossible
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix
