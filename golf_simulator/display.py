"""
display.py
----------
Renders the live camera feed with:
  - MediaPipe skeleton overlay (joints + bones)
  - Swing phase banner
  - HUD: hole info, distance remaining, stroke count
  - Shot result popup (3-second display)
  - End-of-hole / scorecard screen
  - Mini shot-trajectory arc drawn after each shot

All drawing uses OpenCV so no additional windowing library is needed
for the core overlay.  The window is opened with cv2.namedWindow and
can optionally go fullscreen on the Pi HDMI output.
"""

import cv2
import numpy as np
import time
from typing import Optional

from body_tracker import CONNECTIONS, LANDMARKS
from swing_analyzer import SwingPhase, SwingResult
from game_engine import GameEngine, ShotResult, HoleScore


# Colour palette (BGR)
COL_GREEN      = (50,  205,  50)
COL_YELLOW     = (0,   220, 220)
COL_RED        = (50,   50, 220)
COL_WHITE      = (255, 255, 255)
COL_BLACK      = (0,     0,   0)
COL_BLUE       = (220, 100,  30)
COL_ORANGE     = (0,   140, 255)
COL_DARK_BG    = (20,   20,  20)
COL_PHASE      = {
    SwingPhase.IDLE:           (180, 180, 180),
    SwingPhase.ADDRESS:        (200, 200,   0),
    SwingPhase.BACKSWING:      (0,  180, 255),
    SwingPhase.TOP:            (0,  255, 200),
    SwingPhase.DOWNSWING:      (0,  140, 255),
    SwingPhase.IMPACT:         (0,   50, 255),
    SwingPhase.FOLLOW_THROUGH: (0,  220, 100),
    SwingPhase.COMPLETE:       (50, 255,  50),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


def put_text_with_shadow(img, text, pos, font=FONT, scale=0.6,
                         color=COL_WHITE, thickness=1):
    """Draw text with a dark shadow for readability over video."""
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, COL_BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y),   font, scale, color,     thickness,   cv2.LINE_AA)


class Display:
    """Manages the OpenCV window and all drawing operations."""

    SHOT_POPUP_DURATION = 3.5   # seconds
    HOLE_COMPLETE_DURATION = 4.0

    def __init__(self, window_name: str = "Pi Golf Simulator",
                 fullscreen: bool = False):
        self._win = window_name
        self._fullscreen = fullscreen
        self._shot_popup_time: float = 0
        self._last_shot: Optional[ShotResult] = None
        self._hole_complete_time: float = 0
        self._shown_holes: int = 0
        self._scorecard_printed: bool = False
        self._last_hole_score: Optional[HoleScore] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        if self._fullscreen:
            cv2.setWindowProperty(
                self._win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

    def close(self):
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Main render call
    # ------------------------------------------------------------------

    def render(self, frame: np.ndarray, keypoints: dict,
               phase: SwingPhase, game: GameEngine,
               swing_result: Optional[SwingResult] = None) -> int:
        """
        Draw everything onto `frame` and display it.

        Returns the cv2.waitKey value (check for 'q' = quit).
        """
        out = frame.copy()

        # 1. Skeleton
        self._draw_skeleton(out, keypoints)

        # 2. Phase banner (top-centre)
        self._draw_phase_banner(out, phase)

        # 3. HUD (top-left)
        if not game.game_over:
            self._draw_hud(out, game)

        # 4. Shot popup
        if self._last_shot and (time.time() - self._shot_popup_time) < self.SHOT_POPUP_DURATION:
            self._draw_shot_popup(out, self._last_shot, swing_result)
        else:
            self._last_shot = None

        # 5. Hole complete banner
        if self._last_hole_score and \
                (time.time() - self._hole_complete_time) < self.HOLE_COMPLETE_DURATION:
            self._draw_hole_complete(out, self._last_hole_score)

        # 6. Game over scorecard overlay
        if game.game_over:
            self._draw_scorecard_overlay(out, game)

        cv2.imshow(self._win, out)
        return cv2.waitKey(1) & 0xFF

    # ------------------------------------------------------------------
    # Trigger methods (called from main loop on events)
    # ------------------------------------------------------------------

    def show_shot_result(self, shot: ShotResult,
                         swing_result: Optional[SwingResult] = None):
        self._last_shot = shot
        self._last_swing = swing_result
        self._shot_popup_time = time.time()

    def show_hole_complete(self, hole_score: HoleScore):
        self._last_hole_score = hole_score
        self._hole_complete_time = time.time()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_skeleton(self, img, keypoints: dict):
        if not keypoints:
            return
        # Draw bones
        for a_name, b_name in CONNECTIONS:
            if a_name in keypoints and b_name in keypoints:
                a = keypoints[a_name]
                b = keypoints[b_name]
                if a[2] > 0.4 and b[2] > 0.4:
                    cv2.line(img, (a[0], a[1]), (b[0], b[1]), COL_GREEN, 2, cv2.LINE_AA)
        # Draw joints
        for name, (x, y, vis) in keypoints.items():
            if vis > 0.4:
                cv2.circle(img, (x, y), 5, COL_YELLOW, -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), 6, COL_BLACK,   1, cv2.LINE_AA)

    def _draw_phase_banner(self, img, phase: SwingPhase):
        h, w = img.shape[:2]
        label = phase.name.replace("_", " ")
        color = COL_PHASE.get(phase, COL_WHITE)
        # Semi-transparent bar
        overlay = img.copy()
        cv2.rectangle(overlay, (w//2 - 120, 8), (w//2 + 120, 38), COL_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        put_text_with_shadow(img, label, (w//2 - 110, 30),
                             font=FONT_BOLD, scale=0.65, color=color, thickness=1)

    def _draw_hud(self, img, game: GameEngine):
        hole  = game.current_hole
        lines = [
            f"Hole {hole.number}  Par {hole.par}  {hole.yards} yds",
            f"Remaining: {game.distance_remaining} yds",
            f"Stroke: {game.stroke_number}",
        ]
        if hole.hazards:
            lines.append(f"Hazard: {hole.hazards[0]}")

        overlay = img.copy()
        cv2.rectangle(overlay, (8, 8), (310, 8 + len(lines)*26 + 10), COL_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        for i, line in enumerate(lines):
            put_text_with_shadow(img, line, (14, 30 + i*26),
                                 scale=0.55, color=COL_WHITE)

    def _draw_shot_popup(self, img, shot: ShotResult,
                         swing: Optional[SwingResult] = None):
        h, w = img.shape[:2]
        lines = [
            f"  SHOT  -  Stroke {shot.stroke_number}  ",
            f"Distance: {shot.distance} yds",
            f"Direction: {'LEFT' if shot.direction < 0 else 'RIGHT'} {abs(shot.direction)} yds",
            f"Lie: {shot.lie.upper()}",
            f"Remaining: {shot.remaining} yds",
        ]
        if shot.penalty:
            lines.append(f"PENALTY: +{shot.penalty} stroke(s)")
        if shot.message:
            lines.append(shot.message)
        if swing:
            lines.append(f"Power: {swing.power_score:.0f}   Accuracy: {swing.accuracy_score:.0f}")

        box_w, box_h = 320, len(lines) * 26 + 20
        x0 = w - box_w - 15
        y0 = 60

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h), COL_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

        header_col = COL_ORANGE if shot.in_hazard else COL_GREEN
        for i, line in enumerate(lines):
            col = header_col if i == 0 else COL_WHITE
            put_text_with_shadow(img, line, (x0+8, y0 + 22 + i*26),
                                 scale=0.55, color=col)

        # Feedback tips (below popup)
        if swing:
            ty = y0 + box_h + 10
            for tip in swing.feedback[:2]:
                put_text_with_shadow(img, f"* {tip}",
                                     (x0+8, ty), scale=0.45, color=COL_YELLOW)
                ty += 20

    def _draw_hole_complete(self, img, hole_score: HoleScore):
        h, w = img.shape[:2]
        text  = f"Hole {hole_score.hole.number} Complete – {hole_score.score_name}"
        color = COL_GREEN if hole_score.relative_to_par <= 0 else COL_ORANGE
        overlay = img.copy()
        cv2.rectangle(overlay, (w//2 - 220, h//2 - 30),
                      (w//2 + 220, h//2 + 30), COL_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        put_text_with_shadow(img, text, (w//2 - 210, h//2 + 8),
                             font=FONT_BOLD, scale=0.75, color=color, thickness=2)

    def _draw_scorecard_overlay(self, img, game: GameEngine):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (w//2 - 270, h//2 - 160),
                      (w//2 + 270, h//2 + 160), COL_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        put_text_with_shadow(img, "GAME OVER  -  SCORECARD",
                             (w//2 - 200, h//2 - 130),
                             font=FONT_BOLD, scale=0.75, color=COL_YELLOW, thickness=2)

        for i, hs in enumerate(game.hole_scores):
            diff  = hs.relative_to_par
            color = COL_GREEN if diff <= 0 else COL_RED
            line  = (f"Hole {hs.hole.number}  Par {hs.hole.par}  "
                     f"{hs.total} strokes  {hs.score_name}")
            put_text_with_shadow(img, line, (w//2 - 240, h//2 - 85 + i*38),
                                 scale=0.6, color=color)

        total_diff = game.total_score - game.total_par
        sign = "+" if total_diff > 0 else ""
        put_text_with_shadow(
            img,
            f"TOTAL: {game.total_score}  ({sign}{total_diff} vs par)",
            (w//2 - 200, h//2 + 120),
            font=FONT_BOLD, scale=0.70,
            color=COL_WHITE, thickness=2,
        )
        put_text_with_shadow(img, "Press Q to quit",
                             (w//2 - 100, h//2 + 148),
                             scale=0.5, color=COL_YELLOW)
