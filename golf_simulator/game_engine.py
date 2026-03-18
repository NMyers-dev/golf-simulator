"""
game_engine.py
--------------
Golf game state: holes, par, scoring, shot simulation.

3-hole course layout (all distances in yards):
  Hole 1 – Par 4, 380 yds, Slight dogleg right
  Hole 2 – Par 3, 155 yds, Island green (accuracy matters)
  Hole 3 – Par 5, 520 yds, Long straight
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from swing_analyzer import SwingResult


# ---------------------------------------------------------------------------
# Course data
# ---------------------------------------------------------------------------

@dataclass
class HoleInfo:
    number:     int
    par:        int
    yards:      int
    name:       str
    green_width: int   # yards (affects how much direction error is OK)
    hazards:    list   = field(default_factory=list)


COURSE: List[HoleInfo] = [
    HoleInfo(1, 4, 380, "Dogleg Right",  30, ["bunker at 220 yards right"]),
    HoleInfo(2, 3, 155, "Island Green",  18, ["water all around green"]),
    HoleInfo(3, 5, 520, "Long Straight", 35, ["rough left", "bunker at 300 yards"]),
]


# ---------------------------------------------------------------------------
# Shot result
# ---------------------------------------------------------------------------

@dataclass
class ShotResult:
    distance:      int         # yards carried
    direction:     int         # yards off-line (negative = left, positive = right)
    remaining:     int         # yards to hole after shot
    lie:           str         # "fairway", "rough", "bunker", "water", "green", "hole"
    stroke_number: int
    in_hazard:     bool = False
    penalty:       int  = 0    # extra strokes
    message:       str  = ""


@dataclass
class HoleScore:
    hole:     HoleInfo
    strokes:  int  = 0
    penalty:  int  = 0

    @property
    def total(self) -> int:
        return self.strokes + self.penalty

    @property
    def relative_to_par(self) -> int:
        return self.total - self.hole.par

    @property
    def score_name(self) -> str:
        diff = self.relative_to_par
        names = {-3: "Albatross", -2: "Eagle", -1: "Birdie",
                  0: "Par", 1: "Bogey", 2: "Double Bogey"}
        return names.get(diff, f"+{diff}" if diff > 0 else str(diff))


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------

class GameEngine:
    """
    Manages the full 3-hole game loop.

    Typical usage:
        game = GameEngine()
        game.start_game()
        while not game.game_over:
            # get swing_result from analyzer
            shot = game.register_shot(swing_result)
            # display shot info
        scorecard = game.scorecard()
    """

    MAX_STROKES_PER_HOLE = 10

    def __init__(self):
        self.hole_scores: List[HoleScore] = []
        self._current_hole_idx: int = 0
        self._current_distance: int = 0   # distance remaining to hole
        self._stroke: int = 0
        self.game_over: bool = False
        self._current_hole_score: Optional[HoleScore] = None
        self.last_shot: Optional[ShotResult] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_game(self):
        self._current_hole_idx = 0
        self.hole_scores = []
        self.game_over = False
        self._start_hole()

    def _start_hole(self):
        hole = COURSE[self._current_hole_idx]
        self._current_distance = hole.yards
        self._stroke = 0
        self._current_hole_score = HoleScore(hole=hole)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_hole(self) -> HoleInfo:
        return COURSE[self._current_hole_idx]

    @property
    def distance_remaining(self) -> int:
        return self._current_distance

    @property
    def stroke_number(self) -> int:
        return self._stroke + 1

    @property
    def total_score(self) -> int:
        return sum(s.total for s in self.hole_scores)

    @property
    def total_par(self) -> int:
        return sum(s.hole.par for s in self.hole_scores)

    # ------------------------------------------------------------------
    # Shot registration
    # ------------------------------------------------------------------

    def register_shot(self, swing: SwingResult) -> ShotResult:
        """
        Convert a SwingResult into a ShotResult and advance game state.
        Returns the ShotResult for display.
        """
        self._stroke += 1
        hole = self.current_hole
        distance  = swing.distance_yards
        direction = swing.direction_offset

        # Cap distance to remaining + slight overshoot
        distance = min(distance, self._current_distance + 20)

        # Determine lie
        lie, penalty, message = self._determine_lie(
            hole, distance, direction, self._current_distance
        )

        remaining = max(0, self._current_distance - distance)

        # Landing on green – check for hole-out
        holed = False
        if lie == "green" and remaining < 5:
            remaining = 0
            lie = "hole"
            holed = True
        elif lie == "hole":
            remaining = 0
            holed = True

        shot = ShotResult(
            distance      = distance,
            direction     = direction,
            remaining     = remaining,
            lie           = lie,
            stroke_number = self._stroke,
            in_hazard     = penalty > 0,
            penalty       = penalty,
            message       = message,
        )
        self.last_shot = shot

        # Update distance
        self._current_distance = remaining
        assert self._current_hole_score is not None
        self._current_hole_score.strokes += 1
        self._current_hole_score.penalty += penalty

        # Check hole completion
        if holed or self._stroke >= self.MAX_STROKES_PER_HOLE:
            if not holed:
                message = "Max strokes reached – hole conceded"
            self._complete_hole()

        return shot

    def _determine_lie(self, hole: HoleInfo, distance: int,
                       direction: int, remaining: int) -> tuple:
        """Return (lie_string, penalty_strokes, message)."""
        off_line = abs(direction)

        # Check if on the green (within 30 yards of hole)
        if distance >= remaining - 5:
            if off_line <= hole.green_width // 2:
                return "green", 0, "On the green!"
            # Island green special case
            if hole.number == 2 and off_line > hole.green_width // 2:
                return "water", 1, "Water hazard! +1 penalty stroke."

        # Hole 1 bunker at 220 yards right
        if hole.number == 1 and 200 <= distance <= 240 and direction > 15:
            return "bunker", 0, "In the bunker at 220 yards!"

        # Hole 3 bunker at 300
        if hole.number == 3 and 280 <= distance <= 320 and direction < -10:
            return "bunker", 0, "Caught the left bunker!"

        # General rough / fairway
        if off_line <= 15:
            return "fairway", 0, "Great shot – fairway!"
        elif off_line <= 30:
            return "rough", 0, "In the rough."
        else:
            # OB or heavy rough
            if off_line > 50:
                return "rough", 1, "Out of bounds! +1 penalty."
            return "rough", 0, "Heavy rough."

    def _complete_hole(self):
        assert self._current_hole_score is not None
        self.hole_scores.append(self._current_hole_score)
        self._current_hole_idx += 1
        if self._current_hole_idx >= len(COURSE):
            self.game_over = True
        else:
            self._start_hole()

    # ------------------------------------------------------------------
    # Scorecard
    # ------------------------------------------------------------------

    def scorecard(self) -> str:
        lines = []
        lines.append("=" * 50)
        lines.append("           SCORECARD – Pi Golf Simulator")
        lines.append("=" * 50)
        lines.append(f"{'Hole':<6} {'Par':<5} {'Yds':<6} {'Strokes':<9} {'Score'}")
        lines.append("-" * 50)
        for hs in self.hole_scores:
            lines.append(
                f"{hs.hole.number:<6} {hs.hole.par:<5} {hs.hole.yards:<6} "
                f"{hs.total:<9} {hs.score_name}"
            )
        lines.append("-" * 50)
        completed_par = sum(s.hole.par for s in self.hole_scores)
        diff = self.total_score - completed_par
        sign = "+" if diff > 0 else ""
        lines.append(
            f"{'TOTAL':<6} {completed_par:<5} {'':6} {self.total_score:<9} "
            f"{sign}{diff}"
        )
        lines.append("=" * 50)
        return "\n".join(lines)
