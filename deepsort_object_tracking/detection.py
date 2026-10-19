import numpy as np


class Detection:
    """
    Represents one detection in one frame.

    bbox:       [x1, y1, x2, y2]
    confidence: float
    feature:    128-dim appearance embedding (numpy array)
    """

    def __init__(self, bbox, confidence, feature):
        self.bbox       = np.asarray(bbox, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature    = np.asarray(feature, dtype=np.float32)

    def to_tlwh(self):
        """Convert [x1,y1,x2,y2] to [top,left,width,height]"""
        ret = self.bbox.copy()
        ret[2] -= ret[0]   # w = x2 - x1
        ret[3] -= ret[1]   # h = y2 - y1
        return ret

    def to_xyah(self):
        """
        Convert to [center_x, center_y, aspect_ratio, height]
        This is the Kalman measurement format
        """
        ret = self.to_tlwh()
        ret[:2] += ret[2:] / 2    # center
        ret[2]  /= ret[3]         # aspect ratio = w/h
        return ret
