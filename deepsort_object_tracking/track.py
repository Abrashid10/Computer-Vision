class TrackState:
    """Track lifecycle states."""
    Tentative = 1    # newly created, not confirmed
    Confirmed = 2    # matched n_init times, real object
    Deleted   = 3    # lost for max_age frames, remove


class Track:
    """
    Single tracked object.
    Maintained by the Tracker class.
    """

    def __init__(self, mean, covariance, track_id,
                 n_init, max_age, feature=None):
        """
        mean:        initial Kalman state (8,)
        covariance:  initial Kalman covariance (8,8)
        track_id:    unique integer ID
        n_init:      frames needed to confirm (3)
        max_age:     frames before deletion (70)
        feature:     initial appearance embedding (128,)
        """
        self.mean           = mean
        self.covariance     = covariance
        self.track_id       = track_id
        self.hits           = 1           # total matches
        self.age            = 1           # total frames
        self.time_since_update = 0        # frames since last match
        self.state          = TrackState.Tentative
        self.features       = []          # appearance gallery

        self._n_init  = n_init
        self._max_age = max_age

        if feature is not None:
            self.features.append(feature)

    def to_tlwh(self):
        """Get [top, left, width, height] from Kalman state."""
        ret    = self.mean[:4].copy()
        ret[2] *= ret[3]       # w = γ × h
        ret[:2] -= ret[2:] / 2 # top-left = center - size/2
        return ret

    def to_tlbr(self):
        """Get [x1, y1, x2, y2]."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Run Kalman predict step."""
        self.mean, self.covariance = kf.predict(
            self.mean, self.covariance
        )
        self.age            += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """
        Run Kalman update step.
        Add detection feature to gallery.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance,
            detection.to_xyah()
        )

        # Add to appearance gallery (keep last 100)
        self.features.append(detection.feature)
        if len(self.features) > 100:
            self.features.pop(0)

        self.hits              += 1
        self.time_since_update  = 0

        # State transitions
        if self.state == TrackState.Tentative and \
           self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Called when track has no matching detection."""
        if self.state == TrackState.Tentative:
            # Tentative tracks deleted immediately if missed
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            # Confirmed tracks deleted after max_age misses
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
