#!/usr/bin/env bash
# Re-run the scene analysis for every scene with its own options (scale, stairs, vegetation).
set -e
cd "$(dirname "$0")"
PY=../tools/sharp-venv/Scripts/python.exe
OV="$HOME/.claude/skills/painting-walk-splat/scripts/overlay_walkable.py"
run() {  # id, splat ply, extra args...
  local id=$1 ply=$2; shift 2
  echo "=== $id"
  $PY -I analyze_scene.py "$ply" "scenes/$id/scene.json" --hotspots "scenes/$id/hotspots.json" "$@" | grep -vE "splats, image"
  $PY -I "$OV" "scenes/$id/scene.json" "scenes/$id/image.png" "scenes/$id/check.jpg" > /dev/null
}
run street scenes/street/raw.ply --stairs-top 600,650,700,656 --stairs-from -14 --stairs-halfwidth 2.0 --landing 1 --corridor-pct 40 --corridor-free --corridor-drift 1.0
run stairs scenes/stairs/out/stairs.ply --max-step 0.22 --eye-height 1.0 --ground-ceiling 30 --exclude-green --stairs-top 720,329,860,333 --stairs-from -11 --stairs-halfwidth 2.0 --landing 1.5
run train  scenes/train/out/train.ply
run beach  scenes/beach/out/beach.ply --eye-height 1.5 --max-step 0.09
run night  scenes/night/out/night.ply --stairs-top 600,540,720,546 --stairs-from -10 --stairs-halfwidth 1.6 --landing 1 --corridor-pct 20
run home   scenes/home/out/home.ply
