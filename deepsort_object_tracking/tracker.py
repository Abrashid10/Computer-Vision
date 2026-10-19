import numpy as np
from kalman_filter import KalmanFilter
from track import Track, TrackState
from linear_assignment import (matching_cascade,
                                min_cost_matching,
                                gate_cost_matrix)
from nn_matching import NearestNeighborDistanceMetric


class Tracker:
    """
    Main DeepSORT tracker.
    Manages all Track objects.
    """

    def __init__(self, metric, max_iou_distance=0.7,
                 max_age=70, n_init=3):
        """
        metric:           NearestNeighborDistanceMetric instance
        max_iou_distance: IoU threshold for second stage matching
        max_age:          delete after this many missed frames
        n_init:           matches needed to confirm track
        """
        self.metric           = metric
        self.max_iou_distance = max_iou_distance
        self.max_age          = max_age
        self.n_init           = n_init

        self.kf      = KalmanFilter()
        self.tracks  = []
        self._next_id = 1

    def predict(self):
        """Propagate all track states one step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """
        Main update. Called every frame.

        detections: list of Detection objects
        """
        # ── Step 1: Cascade Matching (appearance) ────────
        matches_a, unmatched_tracks_a, unmatched_dets_a = \
            self._match(detections)

        # ── Step 2: IoU Matching for remaining tracks ─────
        # Unconfirmed tracks + confirmed tracks missed once
        iou_track_candidates = [
            i for i in unmatched_tracks_a
            if self.tracks[i].time_since_update == 1
        ]
        unmatched_tracks_b = [
            i for i in unmatched_tracks_a
            if self.tracks[i].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b2, unmatched_dets_b = \
            min_cost_matching(
                self._iou_cost,
                self.max_iou_distance,
                self.tracks, detections,
                iou_track_candidates, unmatched_dets_a
            )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_b +
                                    unmatched_tracks_b2))

        # ── Step 3: Update matched tracks ─────────────────
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[det_idx]
            )

        # ── Step 4: Mark missed tracks ────────────────────
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # ── Step 5: Create new tracks ─────────────────────
        for det_idx in unmatched_dets_b:
            self._initiate_track(detections[det_idx])

        # ── Step 6: Delete dead tracks ────────────────────
        self.tracks = [t for t in self.tracks
                       if not t.is_deleted()]

        # ── Step 7: Update appearance gallery ────────────
        active_targets = [t.track_id for t in self.tracks
                          if t.is_confirmed()]
        features       = []
        targets        = []

        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets  += [track.track_id] * len(track.features)
            track.features = []   # clear after storing

        self.metric.partial_fit(
            np.array(features),
            np.array(targets),
            active_targets
        )

    def _match(self, detections):
        """
        Cascade matching using appearance features + Mahalanobis gating.
        """
        def gated_metric(tracks, dets, track_indices, det_indices):
            # Get detection features
            features = np.array([dets[i].feature for i in det_indices])
            # Get track IDs for gallery lookup
            targets  = np.array([tracks[i].track_id for i in track_indices])

            # Appearance cost matrix (cosine distance)
            cost_matrix = self.metric.distance(features, targets)

            # Apply Mahalanobis gate
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets,
                track_indices, det_indices
            )
            return cost_matrix

        # Only confirmed tracks go into cascade matching
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed()
        ]
        # Unconfirmed tracks go straight to IoU matching
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks)
            if not t.is_confirmed()
        ]

        # Cascade matching for confirmed tracks
        matches_a, unmatched_tracks_a, unmatched_dets_a = \
            matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks, detections,
                confirmed_tracks
            )

        # IoU matching for unconfirmed + recently lost confirmed
        iou_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]

        matches_b, unmatched_tracks_b, unmatched_dets_b = \
            min_cost_matching(
                self._iou_cost,
                self.max_iou_distance,
                self.tracks, detections,
                iou_candidates, unmatched_dets_a
            )

        matches = matches_a + matches_b
        unmatched_tracks = list(
            set(unmatched_tracks_a) - {k for k, _ in matches_b}
        ) + unmatched_tracks_b

        return matches, unmatched_tracks, unmatched_dets_b

    def _iou_cost(self, tracks, detections,
                  track_indices, detection_indices):
        """
        IoU-based cost matrix for second stage matching.
        1 - IoU so that lower = better match.
        """
        cost_matrix = np.zeros((len(track_indices),
                                len(detection_indices)))

        for row, t_idx in enumerate(track_indices):
            if tracks[t_idx].time_since_update > 1:
                cost_matrix[row, :] = 1e5
                continue

            bbox = tracks[t_idx].to_tlbr()

            for col, d_idx in enumerate(detection_indices):
                det_bbox = detections[d_idx].bbox
                cost_matrix[row, col] = 1 - self._iou(bbox, det_bbox)

        return cost_matrix

    def _iou(self, bbox1, bbox2):
        """IoU between two [x1,y1,x2,y2] boxes."""
        xx1 = max(bbox1[0], bbox2[0])
        yy1 = max(bbox1[1], bbox2[1])
        xx2 = min(bbox1[2], bbox2[2])
        yy2 = min(bbox1[3], bbox2[3])

        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        intersection = w * h

        area1 = ((bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1]))
        area2 = ((bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1]))
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _initiate_track(self, detection):
        """Create a new track from an unmatched detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance,
            self._next_id,
            self.n_init,
            self.max_age,
            detection.feature
        ))
        self._next_id += 1
