"""
body_tracker.py
---------------
MediaPipe Pose wrapper for the Raspberry Pi 5 AI Camera.
Captures frames via picamera2 and runs pose estimation.
Falls back to standard OpenCV VideoCapture if picamera2 is unavailable
(useful for development on non-Pi hardware).
"""

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

import mediapipe as mp

# Landmark indices we care about for golf swing analysis
LANDMARKS = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}

CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]


class BodyTracker:
    """
    Wraps picamera2 + MediaPipe Pose.

    Usage:
        tracker = BodyTracker(width=640, height=480, fps=30)
        tracker.start()
        while True:
            frame, keypoints = tracker.get_frame()
            if frame is None:
                break
            # keypoints: dict[str, (x_px, y_px, visibility)] or empty
        tracker.stop()
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6):
        self.width = width
        self.height = height
        self.fps = fps

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._camera = None
        self._cap = None  # fallback OpenCV capture

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Open the camera."""
        if PICAMERA2_AVAILABLE:
            self._camera = Picamera2()
            config = self._camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self._camera.configure(config)
            self._camera.start()
        else:
            print("[BodyTracker] picamera2 not available – using OpenCV VideoCapture(0)")
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def stop(self):
        """Release camera resources."""
        self._pose.close()
        if self._camera:
            self._camera.stop()
        if self._cap:
            self._cap.release()

    # ------------------------------------------------------------------
    # Frame capture + pose estimation
    # ------------------------------------------------------------------

    def get_frame(self):
        """
        Capture one frame and run MediaPipe pose estimation.

        Returns
        -------
        frame : np.ndarray (BGR, HxWx3) or None on error
        keypoints : dict[str -> (x_px, y_px, visibility)]
        raw_landmarks : mediapipe NormalizedLandmarkList or None
        """
        if self._camera:
            frame_rgb = self._camera.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self._cap.read()
            if not ret:
                return None, {}, None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._pose.process(frame_rgb)

        keypoints = {}
        raw_landmarks = None

        if results.pose_landmarks:
            raw_landmarks = results.pose_landmarks
            h, w = frame.shape[:2]
            for name, idx in LANDMARKS.items():
                lm = results.pose_landmarks.landmark[idx]
                keypoints[name] = (int(lm.x * w), int(lm.y * h), lm.visibility)

        return frame, keypoints, raw_landmarks

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def angle_between(a, b, c):
        """
        Compute the angle at point B formed by vectors BA and BC.

        Parameters
        ----------
        a, b, c : (x, y, *) tuples
        Returns angle in degrees [0, 180].
        """
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
