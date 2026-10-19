import numpy as np
from filterpy.kalman import KalmanFilter
from utils import (convert_bbox_to_z,
                   convert_x_to_bbox,
                   associate_detections_to_trackers)


class KalmanBoxTracker:
    """
    Represents one tracked object using a Kalman Filter.
    State: [u, v, s, r, u_dot, v_dot, s_dot]
    """
    count = 0  # class variable — global ID counter

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2] initial detection
        """
        # ── Kalman Filter Setup ──────────────────────────
        # dim_x = 7 (state size)
        # dim_z = 4 (measurement size: u,v,s,r)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix F
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],   # u  = u  + u_dot
            [0,1,0,0,0,1,0],   # v  = v  + v_dot
            [0,0,1,0,0,0,1],   # s  = s  + s_dot
            [0,0,0,1,0,0,0],   # r  = r  (constant)
            [0,0,0,0,1,0,0],   # u_dot constant
            [0,0,0,0,0,1,0],   # v_dot constant
            [0,0,0,0,0,0,1],   # s_dot constant
        ], dtype=float)

        # Measurement matrix H
        # We observe u,v,s,r but NOT velocities
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        # Measurement noise R
        # How much do we trust the detector?
        self.kf.R[2:, 2:] *= 10.  # less trust in scale/ratio

        # Initial uncertainty P
        # High uncertainty for velocities (we don't know them yet)
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P        *= 10.

        # Process noise Q
        # How much can state change unexpectedly?
        self.kf.Q[-1, -1] *= 0.01   # s_dot changes slowly
        self.kf.Q[4:, 4:] *= 0.01   # velocities change slowly

        # Initialize state with first detection
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        # Track metadata
        self.time_since_update = 0    # frames since last matched detection
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []             # predicted box history
        self.hits = 0                 # total times matched
        self.hit_streak = 0           # consecutive matches
        self.age = 0                  # total frames alive

    def update(self, bbox):
        """
        Called when this track is matched to a detection.
        Runs Kalman UPDATE step.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Runs Kalman PREDICT step.
        Called every frame before matching.
        """
        # Prevent negative scale
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0    # streak broken if not matched

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]    # return predicted bbox

    def get_state(self):
        """Return current state as [x1,y1,x2,y2]"""
        return convert_x_to_bbox(self.kf.x)


class SORT:
    """
    Main SORT tracker.
    Manages all KalmanBoxTrackers.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        max_age:       delete track after this many unmatched frames
        min_hits:      confirm track after this many consecutive matches
        iou_threshold: minimum IoU for valid match
        """
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []       # list of KalmanBoxTrackers
        self.frame_count   = 0

    def update(self, detections):
        """
        Main update function. Called every frame.

        detections: numpy array (N, 5) [x1,y1,x2,y2,score]
                    or (N, 4) [x1,y1,x2,y2]

        Returns: numpy array (M, 5) [x1,y1,x2,y2,track_id]
        """
        self.frame_count += 1

        # ── Step 1: Predict new positions for all trackers ──
        predicted_boxes = np.zeros((len(self.trackers), 5))
        to_delete = []

        for t, tracker in enumerate(self.trackers):
            pred = tracker.predict()[0]   # [x1,y1,x2,y2]
            predicted_boxes[t] = [*pred, 0]

            # Mark tracker if prediction is invalid (NaN)
            if np.any(np.isnan(pred)):
                to_delete.append(t)

        # Remove invalid trackers
        predicted_boxes = np.ma.compress_rows(
            np.ma.masked_invalid(predicted_boxes)
        )
        for t in reversed(to_delete):
            self.trackers.pop(t)

        # ── Step 2: Match predictions to detections ──
        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(
                detections,
                predicted_boxes,
                self.iou_threshold
            )

        # ── Step 3: Update matched trackers ──
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        # ── Step 4: Create new trackers for unmatched detections ──
        for i in unmatched_dets:
            new_tracker = KalmanBoxTracker(detections[i])
            self.trackers.append(new_tracker)

        # ── Step 5: Build output + delete old trackers ──
        results = []
        for tracker in reversed(self.trackers):

            # Only output if:
            # 1. Recently updated (time_since_update < max_age)
            # 2. Confirmed (hit_streak >= min_hits OR early frames)
            if (tracker.time_since_update < self.max_age) and \
               (tracker.hit_streak >= self.min_hits or
                self.frame_count <= self.min_hits):

                box = tracker.get_state()[0]  # [x1,y1,x2,y2]
                results.append(
                    [*box, tracker.id + 1]    # +1 so IDs start at 1
                )

        # Delete dead trackers
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        if len(results) > 0:
            return np.array(results)
        else:
            return np.empty((0, 5))
