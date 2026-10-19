import cv2
import numpy as np
from sort import SORT
from detector import YOLOv3Detector

# ── Colors for visualization ──────────────────────────────
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3))

def draw_tracks(frame, tracks):
    """Draw bounding boxes and IDs on frame."""
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        track_id = int(track_id)

        # Unique color per ID
        color = COLORS[track_id % len(COLORS)].tolist()

        # Draw box
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        # Draw ID label
        label = f"ID: {track_id}"
        cv2.putText(
            frame, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )
    return frame


def run_tracking(video_path, output_path=None):
    # ── Initialize ────────────────────────────────────────
    detector = YOLOv3Detector(
        conf_threshold=0.5,
        classes=['person', 'car', 'truck', 'bus']
    )

    tracker = SORT(
        max_age=1,        # from paper
        min_hits=3,       # from paper
        iou_threshold=0.3 # from paper
    )

    cap = cv2.VideoCapture(video_path)

    # Output video writer
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
        detections = detector.detect(frame)
        # detections: (N, 5) [x1,y1,x2,y2,conf]

        # ── Step 2: Track ─────────────────────────────────
        tracks = tracker.update(detections)
        # tracks: (M, 5) [x1,y1,x2,y2,track_id]

        # ── Step 3: Visualize ─────────────────────────────
        frame = draw_tracks(frame, tracks)

        # Show detection count and track count
        cv2.putText(
            frame,
            f"Detections: {len(detections)}  Tracks: {len(tracks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2
        )

        cv2.imshow("SORT Tracking", frame)

        if output_path:
            writer.write(frame)

        # Press q to quit
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
##remember to do this at the start 
#pip install filterpy scipy numpy opencv-python torch torchvision if required especially the opencv others are inbuilt in
