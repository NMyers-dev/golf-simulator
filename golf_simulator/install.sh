#!/usr/bin/env bash
# install.sh – Set up the Pi Golf Simulator on Raspberry Pi 5 with AI Camera
# Run as: bash install.sh

set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Pi Golf Simulator – Dependency Installer"
echo "============================================"

# ── 1. System packages ──────────────────────────────────────────────────────
echo "[1/4] Installing system packages…"
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libharfbuzz0b \
    libwebp7 \
    libtiff6 \
    libjasper-dev \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    v4l-utils \
    libcamera-tools \
    python3-libcamera \
    python3-picamera2 \
    --no-install-recommends

# ── 2. Python virtual environment ───────────────────────────────────────────
echo "[2/4] Creating Python virtual environment at $PROJ_DIR/venv…"
python3 -m venv --system-site-packages "$PROJ_DIR/venv"
source "$PROJ_DIR/venv/bin/activate"

# ── 3. Python packages ───────────────────────────────────────────────────────
echo "[3/4] Installing Python packages…"
pip install --upgrade pip wheel
# MediaPipe – use the official ARM build for Pi
pip install mediapipe
pip install numpy pygame

# Note: opencv-python is provided by system python3-opencv above
# (avoids heavy recompilation).  If you prefer a pip version:
# pip install opencv-python-headless

# ── 4. Camera / udev permissions ─────────────────────────────────────────────
echo "[4/4] Setting up camera permissions…"
# Ensure the user is in the 'video' group (for /dev/video*)
sudo usermod -aG video "$USER" || true

# Enable the camera in /boot/config.txt if not already present
BOOT_CFG="/boot/firmware/config.txt"
if [ ! -f "$BOOT_CFG" ]; then
    BOOT_CFG="/boot/config.txt"
fi
if ! grep -q "camera_auto_detect=1" "$BOOT_CFG" 2>/dev/null; then
    echo "camera_auto_detect=1" | sudo tee -a "$BOOT_CFG"
    echo "  Added camera_auto_detect=1 to $BOOT_CFG"
fi

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  Activate the venv:"
echo "    source $PROJ_DIR/venv/bin/activate"
echo ""
echo "  Run the simulator:"
echo "    python main.py"
echo ""
echo "  Flags:"
echo "    --fullscreen        HDMI fullscreen mode"
echo "    --width 1280 --height 720  Higher resolution"
echo "    --no-camera         Use webcam (testing on PC)"
echo ""
echo "  Controls during play:"
echo "    Q / ESC  – Quit"
echo "    R        – Reset current swing"
echo "    S        – Skip current hole"
echo "============================================"
