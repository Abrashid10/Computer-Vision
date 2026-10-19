
import numpy as np

def iou(bb_test, bb_gt):
    """
    Compute IoU between two boxes.
    Boxes in format [x1, y1, x2, y2]
    """
    # Find intersection rectangle
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    # Intersection area
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h

    # Union area
    area_test = ((bb_test[2] - bb_test[0]) *
                 (bb_test[3] - bb_test[1]))
    area_gt   = ((bb_gt[2]   - bb_gt[0])   *
                 (bb_gt[3]   - bb_gt[1]))
    union = area_test + area_gt - intersection

    return intersection / union


def convert_bbox_to_z(bbox):
    """
    Convert [x1,y1,x2,y2] to Kalman state [u,v,s,r]

    u = center x
    v = center y
    s = scale (area)
    r = aspect ratio (w/h)
    """
    w = bbox[2] - bbox[0]   # width
    h = bbox[3] - bbox[1]   # height
    u = bbox[0] + w / 2     # center x
    v = bbox[1] + h / 2     # center y
    s = w * h                # area
    r = w / float(h)         # aspect ratio

    return np.array([u, v, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert Kalman state [u,v,s,r] back to [x1,y1,x2,y2]
    """
    w = np.sqrt(x[2] * x[3])      # width  = sqrt(s × r)
    h = x[2] / w                   # height = s / w

    box = [
        x[0] - w / 2,   # x1
        x[1] - h / 2,   # y1
        x[0] + w / 2,   # x2
        x[1] + h / 2    # y2
    ]

    if score is None:
        return np.array(box).reshape((1, 4))
    else:
        return np.array(box + [score]).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Hungarian algorithm assignment.

    detections:  (N, 5) array [x1,y1,x2,y2,score]
    trackers:    (M, 5) array [x1,y1,x2,y2,track_id]

    Returns:
        matches:         matched (detection_idx, tracker_idx) pairs
        unmatched_dets:  detections with no match
        unmatched_trks:  trackers with no match
    """
    from scipy.optimize import linear_sum_assignment

    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int)
        )

    # Build IoU matrix (cost matrix)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Hungarian algorithm — maximize IoU (minimize negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    # Filter out matches below IoU threshold
    unmatched_dets = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_trks = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            # IoU too low → treat as unmatched
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return (
        matches,
        np.array(unmatched_dets),
        np.array(unmatched_trks)
    )
