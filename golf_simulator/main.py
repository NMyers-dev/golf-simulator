"""
main.py
-------
Entry point for the Raspberry Pi 5 Golf Simulator.

Run with:
    python main.py

Optional flags:
    --width  INT     Camera width  (default 640)
    --height INT     Camera height (default 480)
    --fps    INT     Camera FPS    (default 30)
    --fullscreen     Open display fullscreen (good for Pi HDMI output)
    --no-camera      Use webcam index 0 instead of picamera2

Controls:
    Q / ESC   Quit at any time
    R         Force-reset current swing (if stuck)
    S         Skip current hole (cheat mode)
"""

import argparse
import sys
import time
import cv2

from body_tracker import BodyTracker
from swing_analyzer import SwingAnalyzer, SwingPhase
from game_engine import GameEngine
from display import Display


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pi Golf Simulator")
    p.add_argument("--width",       type=int, default=640)
    p.add_argument("--height",      type=int, default=480)
    p.add_argument("--fps",         type=int, default=30)
    p.add_argument("--fullscreen",  action="store_true")
    p.add_argument("--no-camera",   action="store_true",
                   help="Force OpenCV VideoCapture instead of picamera2")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.no_camera:
        import body_tracker as bt
        bt.PICAMERA2_AVAILABLE = False

    tracker  = BodyTracker(width=args.width, height=args.height, fps=args.fps)
    analyzer = SwingAnalyzer()
    game     = GameEngine()
    display  = Display(fullscreen=args.fullscreen)

    print("[Pi Golf Simulator] Starting…")
    tracker.start()
    display.open()
    game.start_game()

    prev_phase   = SwingPhase.IDLE
    last_swing   = None
    waiting_msg_shown = False

    print(f"[Game] Hole 1 – {game.current_hole.name}  "
          f"{game.current_hole.yards} yds  Par {game.current_hole.par}")

    try:
        while True:
            # ----------------------------------------------------------
            # 1. Capture frame + detect pose
            # ----------------------------------------------------------
            frame, keypoints, _ = tracker.get_frame()
            if frame is None:
                print("[Error] Could not read camera frame. Exiting.")
                break

            # ----------------------------------------------------------
            # 2. Update swing state machine
            # ----------------------------------------------------------
            if not game.game_over:
                phase = analyzer.update(keypoints)

                # Log phase transitions
                if phase != prev_phase:
                    print(f"  [Swing] {prev_phase.name} -> {phase.name}")
                    prev_phase = phase

                # Swing complete – register shot
                if analyzer.swing_complete and analyzer.swing_result is not None:
                    result = analyzer.swing_result
                    last_swing = result

                    print(f"\n  [Swing Result]")
                    print(f"    Power:    {result.power_score:.1f}")
                    print(f"    Accuracy: {result.accuracy_score:.1f}")
                    print(f"    Hip rot:  {result.hip_rotation:.1f} deg")
                    print(f"    Arm ext:  {result.arm_extension:.1f} deg")
                    print(f"    Tempo:    {result.tempo_ratio:.2f}:1")
                    for tip in result.feedback:
                        print(f"    > {tip}")

                    shot = game.register_shot(result)
                    display.show_shot_result(shot, result)

                    print(f"\n  [Shot]  {shot.distance} yds  "
                          f"{'L' if shot.direction < 0 else 'R'}{abs(shot.direction)} yds  "
                          f"lie={shot.lie}  remaining={shot.remaining} yds")
                    if shot.penalty:
                        print(f"  [Penalty] +{shot.penalty}")

                    # Check if the hole just finished
                    # (game._current_hole_idx advances inside register_shot)
                    hole_scores = game.hole_scores
                    if hole_scores:
                        last_hs = hole_scores[-1]
                        # Just completed a hole?  The hole we were on has advanced.
                        if (not game.game_over and
                                game.current_hole.number != last_hs.hole.number + 1
                                and len(hole_scores) > 0):
                            pass  # still same hole
                        # Always show hole-complete banner when last_hs is newly added
                        if len(hole_scores) > getattr(display, "_shown_holes", 0):
                            display.show_hole_complete(last_hs)
                            display._shown_holes = len(hole_scores)
                            print(f"\n  [Hole {last_hs.hole.number}] "
                                  f"{last_hs.score_name}  "
                                  f"({last_hs.total} strokes)")
                            if not game.game_over:
                                print(f"\n[Game] Hole {game.current_hole.number} – "
                                      f"{game.current_hole.name}  "
                                      f"{game.current_hole.yards} yds  "
                                      f"Par {game.current_hole.par}")

                    # Reset analyzer for next swing
                    analyzer.reset()
                    waiting_msg_shown = False

                elif phase == SwingPhase.ADDRESS and not waiting_msg_shown:
                    print("  [Ready] Step into frame and take your swing…")
                    waiting_msg_shown = True

            else:
                phase = SwingPhase.COMPLETE
                if not getattr(display, "_scorecard_printed", False):
                    print("\n" + game.scorecard())
                    display._scorecard_printed = True

            # ----------------------------------------------------------
            # 3. Render
            # ----------------------------------------------------------
            key = display.render(frame, keypoints, phase, game, last_swing)

            # ----------------------------------------------------------
            # 4. Handle key input
            # ----------------------------------------------------------
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                print("[Quit] Bye!")
                break
            elif key in (ord("r"), ord("R")):
                print("[Reset] Swing reset.")
                analyzer.reset()
                waiting_msg_shown = False
            elif key in (ord("s"), ord("S")) and not game.game_over:
                print("[Skip] Hole skipped.")
                # Force hole completion with a bad score
                from swing_analyzer import SwingResult as SR
                dummy = SR(power_score=10, accuracy_score=10,
                           spine_tilt=35, hip_rotation=5,
                           arm_extension=120, tempo_ratio=3.0)
                # Use max remaining strokes
                for _ in range(GameEngine.MAX_STROKES_PER_HOLE):
                    if game.game_over:
                        break
                    game.register_shot(dummy)
                analyzer.reset()

    finally:
        tracker.stop()
        display.close()
        print("[Pi Golf Simulator] Shutdown complete.")


if __name__ == "__main__":
    main()
