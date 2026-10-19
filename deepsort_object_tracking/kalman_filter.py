import numpy as np
import scipy.linalg


# Chi-squared distribution 95th percentile
# 4 degrees of freedom (measurement dim)
# Used for Mahalanobis gating threshold
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,   # ← we use this (dim_z=4)
    5: 11.0705,
    6: 12.5916,
    7: 14.0671,
    8: 15.5073,
    9: 16.9190
}


class KalmanFilter:
    """
    Kalman filter for bounding box tracking.

    State:       [u, v, γ, h, u̇, v̇, γ̇, ḣ]  (8-dim)
    Measurement: [u, v, γ, h]               (4-dim)
    """

    def __init__(self):
        ndim = 4   # measurement dims
        dt   = 1   # time step (one frame)

        # ── State Transition Matrix F (8×8) ──────────────
        # x_new = F @ x_old
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # Result:
        # [1,0,0,0, 1,0,0,0]   u  = u  + u_dot
        # [0,1,0,0, 0,1,0,0]   v  = v  + v_dot
        # [0,0,1,0, 0,0,1,0]   γ  = γ  + γ_dot
        # [0,0,0,1, 0,0,0,1]   h  = h  + h_dot
        # [0,0,0,0, 1,0,0,0]   u_dot constant
        # [0,0,0,0, 0,1,0,0]   v_dot constant
        # [0,0,0,0, 0,0,1,0]   γ_dot constant
        # [0,0,0,0, 0,0,0,1]   h_dot constant

        # ── Measurement Matrix H (4×8) ────────────────────
        # z = H @ x
        self._update_mat = np.eye(ndim, 2 * ndim)
        # Result:
        # [1,0,0,0, 0,0,0,0]   observe u
        # [0,1,0,0, 0,0,0,0]   observe v
        # [0,0,1,0, 0,0,0,0]   observe γ
        # [0,0,0,1, 0,0,0,0]   observe h

        # ── Noise Scale Factors ───────────────────────────
        # These scale the identity matrices for Q and R
        self._std_weight_position = 1. / 20    # position noise
        self._std_weight_velocity = 1. / 160   # velocity noise

    def initiate(self, measurement):
        """
        Create initial state from first measurement.

        measurement: [u, v, γ, h]
        Returns: (mean, covariance) — initial state estimate
        """
        mean_pos = measurement                    # [u, v, γ, h]
        mean_vel = np.zeros_like(mean_pos)        # [0, 0, 0, 0]
        mean     = np.r_[mean_pos, mean_vel]      # [u,v,γ,h,0,0,0,0]

        # Initial uncertainty — high for velocities
        std = [
            2 * self._std_weight_position * measurement[3],   # u
            2 * self._std_weight_position * measurement[3],   # v
            1e-2,                                              # γ
            2 * self._std_weight_position * measurement[3],   # h
            10 * self._std_weight_velocity * measurement[3],  # u_dot
            10 * self._std_weight_velocity * measurement[3],  # v_dot
            1e-5,                                              # γ_dot
            10 * self._std_weight_velocity * measurement[3],  # h_dot
        ]

        covariance = np.diag(np.square(std))   # P = diag(std²)
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Kalman PREDICT step.

        x̂ = F @ x
        P̂ = F @ P @ Fᵀ + Q

        Returns: (predicted_mean, predicted_covariance)
        """
        # Process noise Q — how much can state change unexpectedly
        std_pos = [
            self._std_weight_position * mean[3],   # u
            self._std_weight_position * mean[3],   # v
            1e-2,                                   # γ
            self._std_weight_position * mean[3],   # h
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],   # u_dot
            self._std_weight_velocity * mean[3],   # v_dot
            1e-5,                                   # γ_dot
            self._std_weight_velocity * mean[3],   # h_dot
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # Q

        # Predict
        mean       = self._motion_mat @ mean                              # F @ x
        covariance = (self._motion_mat @ covariance @
                      self._motion_mat.T + motion_cov)                    # F@P@Fᵀ + Q

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state to measurement space.

        μ_z = H @ x
        S   = H @ P @ Hᵀ + R

        Returns: (projected_mean, innovation_covariance)
        """
        # Measurement noise R
        std = [
            self._std_weight_position * mean[3],   # u
            self._std_weight_position * mean[3],   # v
            1e-1,                                   # γ
            self._std_weight_position * mean[3],   # h
        ]
        innovation_cov = np.diag(np.square(std))   # R

        mean       = self._update_mat @ mean                              # H @ x
        covariance = (self._update_mat @ covariance @
                      self._update_mat.T + innovation_cov)                # H@P@Hᵀ + R

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Kalman UPDATE step.

        y = z - H @ x̂              (innovation)
        S = H @ P̂ @ Hᵀ + R         (innovation covariance)
        K = P̂ @ Hᵀ @ S⁻¹           (Kalman gain)
        x = x̂ + K @ y              (updated mean)
        P = (I - K @ H) @ P̂        (updated covariance)

        Returns: (updated_mean, updated_covariance)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Solve for Kalman gain: K = P @ Hᵀ @ S⁻¹
        # Equivalent to solving: Sᵀ @ Kᵀ = (P @ Hᵀ)ᵀ
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False
        ).T
        # kalman_gain = K  shape (8, 4)

        # Innovation
        innovation = measurement - projected_mean   # y = z - H@x̂

        # Update
        new_mean       = mean + innovation @ kalman_gain.T        # x = x̂ + K@y
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """
        Compute Mahalanobis distance between state and measurements.
        Used to gate unlikely matches.

        Returns: array of distances, one per measurement
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            # Only use u,v (ignore γ, h)
            projected_mean = projected_mean[:2]
            projected_cov  = projected_cov[:2, :2]
            measurements   = measurements[:, :2]

        # Cholesky decomposition for efficient computation
        cholesky_factor = np.linalg.cholesky(projected_cov)

        # d = measurement - projected_mean
        d = measurements - projected_mean

        # Solve: L @ z = dᵀ  →  z = L⁻¹ @ dᵀ
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T,
            lower=True, check_finite=False,
            overwrite_b=True
        )

        # Mahalanobis distance = sum(z²) per measurement
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
