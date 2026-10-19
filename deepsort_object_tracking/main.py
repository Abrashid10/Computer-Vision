import cv2
import numpy as np
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from appearance.extractor import FeatureExtractor
from detector import YOLOv3Detector


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(300, 3))


def draw_tracks(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue

        bbox     = track.to_tlbr()
        track_id = track.track_id
        x1,y1,x2,y2 = [int(v) for v in bbox]

        color = COLORS[track_id % len(COLORS)].tolist()
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            frame, f"ID:{track_id}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )
    return frame


def run_tracking(video_path, output_path=None, model_path=None):

    # ── Initialize components ─────────────────────────────
    detector  = YOLOv3Detector(conf_threshold=0.5,
                                classes=['person','car','truck'])

    extractor = FeatureExtractor(model_path=model_path)

    metric    = NearestNeighborDistanceMetric(
        matching_threshold=0.2,   # cosine distance threshold
        budget=100                # max features per track gallery
    )

    tracker   = Tracker(
        metric=metric,
        max_iou_distance=0.7,
        max_age=70,               # from paper
        n_init=3                  # from paper
    )

    cap = cv2.VideoCapture(video_path)

    if output_path:
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (width, height)
        )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(f"Frame {frame_idx}")

        # ── Step 1: Detect ────────────────────────────────
        raw_dets = detector.detect(frame)
        # raw_dets: (N, 5) [x1,y1,x2,y2,conf]

        # ── Step 2: Extract appearance features ──────────
        if len(raw_dets) > 0:
            bboxes   = raw_dets[:, :4]
            features = extractor(frame, bboxes)   # (N, 128)
        else:
            features = []

        # ── Step 3: Build Detection objects ───────────────
        detections = []
        for i, det in enumerate(raw_dets):
            x1,y1,x2,y2,conf = det
            detections.append(Detection(
                bbox=[x1,y1,x2,y2],
                confidence=conf,
                feature=features[i]
            ))

        # ── Step 4: Kalman predict ────────────────────────
        tracker.predict()

        # ── Step 5: Update tracker ────────────────────────
        tracker.update(detections)

        # ── Step 6: Visualize ─────────────────────────────
        frame = draw_tracks(frame, tracker.tracks)

        confirmed = sum(1 for t in tracker.tracks if t.is_confirmed())
        cv2.putText(
            frame,
            f"Detections:{len(detections)}  Tracks:{confirmed}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2
        )

        cv2.imshow("DeepSORT", frame)
        if output_path:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for video_path in input_folder.glob("*.mp4"):
        output_path = output_folder / video_path.name

        run_tracking(
            video_path=str(video_path),
            output_path=str(output_path),
            model_path=None
        )
