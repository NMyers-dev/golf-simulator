"""
swing_analyzer.py
-----------------
Detects and scores a golf swing from a stream of pose keypoints.

Swing state machine:
  IDLE  ->  BACKSWING  ->  IMPACT  ->  FOLLOW_THROUGH  ->  COMPLETE

Metrics extracted:
  - spine_tilt      : forward lean angle at address / impact
  - hip_rotation    : degrees hips rotate between address and impact
  - arm_extension   : how straight the lead arm is at top of backswing
  - tempo           : backswing frames / follow-through frames (ideal ~3:1)
  - swing_plane     : consistency of wrist arc height
  - power_score     : composite 0-100 based on hip speed + arm extension
  - accuracy_score  : composite 0-100 based on spine tilt + swing plane
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from body_tracker import BodyTracker


class SwingPhase(Enum):
    IDLE           = auto()
    ADDRESS        = auto()
    BACKSWING      = auto()
    TOP            = auto()
    DOWNSWING      = auto()
    IMPACT         = auto()
    FOLLOW_THROUGH = auto()
    COMPLETE       = auto()


@dataclass
class SwingResult:
    power_score:    float = 0.0   # 0–100
    accuracy_score: float = 0.0   # 0–100
    spine_tilt:     float = 0.0   # degrees
    hip_rotation:   float = 0.0   # degrees
    arm_extension:  float = 0.0   # degrees (180 = fully straight)
    tempo_ratio:    float = 0.0   # backswing / follow-through frames
    feedback:       list  = field(default_factory=list)

    @property
    def distance_yards(self) -> int:
        """Rough simulated distance based on power 0-300 yds."""
        return int(self.power_score * 3.0)

    @property
    def direction_offset(self) -> int:
        """
        Simulated direction offset in yards (left/right of target).
        Perfect accuracy = 0, poor accuracy = large offset.
        """
        # Accuracy 100 => 0 offset; accuracy 0 => +-50 yards random
        max_offset = int((100 - self.accuracy_score) * 0.5)
        if max_offset == 0:
            return 0
        return int(np.random.randint(-max_offset, max_offset + 1))


class SwingAnalyzer:
    """
    Feed keypoint dicts frame-by-frame via `update()`.
    When a swing completes, `swing_result` is populated and
    `swing_complete` is True until you call `reset()`.
    """

    # Thresholds
    WRIST_MOVEMENT_THRESHOLD = 8     # px movement to start detecting swing
    BACKSWING_WRIST_RISE     = 0.15  # fraction of frame height wrist must rise
    IMPACT_VELOCITY_THRESH   = 15    # px/frame wrist speed at impact
    MIN_SWING_FRAMES         = 10
    HISTORY                  = 60    # frames of history to keep

    def __init__(self):
        self._history: deque = deque(maxlen=self.HISTORY)
        self._phase = SwingPhase.IDLE
        self._address_kp  = None
        self._top_kp      = None
        self._impact_kp   = None
        self._backswing_frames = 0
        self._followthrough_frames = 0
        self.swing_complete = False
        self.swing_result: SwingResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, keypoints: dict) -> SwingPhase:
        """
        Process one frame of keypoints.
        Returns the current SwingPhase.
        """
        self._history.append(keypoints)

        if not keypoints:
            return self._phase

        if self._phase == SwingPhase.IDLE:
            self._phase = SwingPhase.ADDRESS
            self._address_kp = keypoints

        elif self._phase == SwingPhase.ADDRESS:
            if self._wrist_moved(keypoints):
                self._phase = SwingPhase.BACKSWING
                self._backswing_frames = 0

        elif self._phase == SwingPhase.BACKSWING:
            self._backswing_frames += 1
            if self._is_at_top(keypoints):
                self._phase = SwingPhase.TOP
                self._top_kp = keypoints

        elif self._phase == SwingPhase.TOP:
            self._phase = SwingPhase.DOWNSWING
            self._followthrough_frames = 0

        elif self._phase == SwingPhase.DOWNSWING:
            if self._is_impact(keypoints):
                self._phase = SwingPhase.IMPACT
                self._impact_kp = keypoints

        elif self._phase == SwingPhase.IMPACT:
            self._phase = SwingPhase.FOLLOW_THROUGH

        elif self._phase == SwingPhase.FOLLOW_THROUGH:
            self._followthrough_frames += 1
            if self._followthrough_frames >= 8:
                self._phase = SwingPhase.COMPLETE
                self._finalise_swing()

        return self._phase

    def reset(self):
        """Reset analyzer for the next swing."""
        self._history.clear()
        self._phase = SwingPhase.IDLE
        self._address_kp = None
        self._top_kp = None
        self._impact_kp = None
        self._backswing_frames = 0
        self._followthrough_frames = 0
        self.swing_complete = False
        self.swing_result = None

    # ------------------------------------------------------------------
    # Internal detection helpers
    # ------------------------------------------------------------------

    def _wrist_moved(self, kp: dict) -> bool:
        if not self._address_kp:
            return False
        for side in ("left_wrist", "right_wrist"):
            if side in kp and side in self._address_kp:
                dx = kp[side][0] - self._address_kp[side][0]
                dy = kp[side][1] - self._address_kp[side][1]
                if np.hypot(dx, dy) > self.WRIST_MOVEMENT_THRESHOLD:
                    return True
        return False

    def _is_at_top(self, kp: dict) -> bool:
        """
        Top of backswing: dominant wrist is above shoulder level
        and has slowed (velocity near zero).
        """
        if len(self._history) < 4:
            return False
        prev = list(self._history)[-4]
        for side in ("right_wrist", "left_wrist"):
            shoulder = side.replace("wrist", "shoulder")
            if side in kp and shoulder in kp and side in prev:
                wrist_y = kp[side][1]
                shoulder_y = kp[shoulder][1]
                # In image coords, smaller y = higher on screen
                if wrist_y < shoulder_y:
                    vx = kp[side][0] - prev[side][0]
                    vy = kp[side][1] - prev[side][1]
                    vel = np.hypot(vx, vy)
                    if vel < 6 and self._backswing_frames >= 5:
                        return True
        return False

    def _is_impact(self, kp: dict) -> bool:
        """Impact: wrist velocity is high and wrist is near hip height."""
        if len(self._history) < 3:
            return False
        prev = list(self._history)[-3]
        for side in ("right_wrist", "left_wrist"):
            hip = side.replace("wrist", "hip")
            if side in kp and hip in kp and side in prev:
                vx = kp[side][0] - prev[side][0]
                vy = kp[side][1] - prev[side][1]
                vel = np.hypot(vx, vy)
                wrist_y = kp[side][1]
                hip_y   = kp[hip][1]
                if vel > self.IMPACT_VELOCITY_THRESH and abs(wrist_y - hip_y) < 80:
                    return True
        return False

    # ------------------------------------------------------------------
    # Score calculation
    # ------------------------------------------------------------------

    def _finalise_swing(self):
        kp_addr   = self._address_kp  or {}
        kp_top    = self._top_kp      or {}
        kp_impact = self._impact_kp   or {}

        spine_tilt    = self._calc_spine_tilt(kp_addr)
        hip_rotation  = self._calc_hip_rotation(kp_addr, kp_impact)
        arm_extension = self._calc_arm_extension(kp_top)
        tempo_ratio   = (self._backswing_frames /
                         max(self._followthrough_frames, 1))

        # --- Power score (0-100) ---
        # Hip rotation ideally 45+ degrees, arm extension ideally 160+
        hip_component  = min(hip_rotation / 60.0, 1.0) * 50
        arm_component  = min(arm_extension / 170.0, 1.0) * 50
        power_score    = hip_component + arm_component

        # --- Accuracy score (0-100) ---
        # Spine tilt ideally 30-40 deg, tempo ideally 2.5-3.5
        spine_score  = max(0, 1 - abs(spine_tilt - 35) / 35) * 50
        tempo_score  = max(0, 1 - abs(tempo_ratio - 3.0) / 3.0) * 50
        accuracy_score = spine_score + tempo_score

        feedback = self._generate_feedback(
            spine_tilt, hip_rotation, arm_extension, tempo_ratio
        )

        self.swing_result = SwingResult(
            power_score    = round(power_score,    1),
            accuracy_score = round(accuracy_score, 1),
            spine_tilt     = round(spine_tilt,     1),
            hip_rotation   = round(hip_rotation,   1),
            arm_extension  = round(arm_extension,  1),
            tempo_ratio    = round(tempo_ratio,    2),
            feedback       = feedback,
        )
        self.swing_complete = True

    def _calc_spine_tilt(self, kp: dict) -> float:
        ls = kp.get("left_shoulder")
        rs = kp.get("right_shoulder")
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        if not all([ls, rs, lh, rh]):
            return 35.0
        assert ls and rs and lh and rh
        mid_shoulder = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
        mid_hip      = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]
        angle = abs(np.degrees(np.arctan2(dx, -dy)))
        return float(angle)

    def _calc_hip_rotation(self, kp_addr: dict, kp_impact: dict) -> float:
        def hip_angle(kp):
            lh = kp.get("left_hip")
            rh = kp.get("right_hip")
            if not lh or not rh:
                return 0.0
            return float(np.degrees(np.arctan2(rh[1]-lh[1], rh[0]-lh[0])))

        return abs(hip_angle(kp_impact) - hip_angle(kp_addr))

    def _calc_arm_extension(self, kp: dict) -> float:
        for side in (("left_shoulder", "left_elbow", "left_wrist"),
                     ("right_shoulder", "right_elbow", "right_wrist")):
            s, e, w = side
            if s in kp and e in kp and w in kp:
                return BodyTracker.angle_between(kp[s], kp[e], kp[w])
        return 140.0

    def _generate_feedback(self, spine, hips, arm, tempo) -> list:
        tips = []
        if spine < 20:
            tips.append("Stand taller – too upright at address")
        elif spine > 50:
            tips.append("Too much forward bend – risk of back injury")
        else:
            tips.append("Good posture!")

        if hips < 30:
            tips.append("Rotate your hips more through the ball")
        elif hips > 80:
            tips.append("Hip rotation is excessive – may cause slice")
        else:
            tips.append("Great hip drive!")

        if arm < 130:
            tips.append("Keep your lead arm straighter at the top")
        else:
            tips.append("Excellent arm extension!")

        if tempo < 1.5:
            tips.append("Slow down your backswing for better tempo")
        elif tempo > 4.5:
            tips.append("Speed up your downswing / follow-through")
        else:
            tips.append(f"Tempo looks good ({tempo:.1f}:1)")

        return tips
