#!/bin/bash
# Runs generate_novel_view_video.py (3 views) for a sequence of episodes,
# stopping only *before* starting a new episode once the cutoff time has
# passed -- never interrupts an episode that has already started.
set -e

CUTOFF_EPOCH=$(date -d "$1" +%s)
shift
BASE_ROOT="/mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__sweep_6cam_newdemos_70"
CKPT="/mnt/cps_persistent1_shared/puneeth/experiments/zeronvs_robot_finetune_sweep/checkpoints/epoch=000001-step=000040000.ckpt"
CONFIG="/home/puneeth/Desktop/VISTA_Data/models/zeronvs_config_robot_finetune.yaml"
COMMON="--checkpoint_path $CKPT --config_path $CONFIG --precomputed_scale 0.52 --guidance_scale 3.0 --ddim_steps 200 --lpips_threshold 0.8"
STATE_FILE="/tmp/claude-1006/-home-puneeth-Desktop-VISTA/d3505877-ed80-468e-9ecf-022087336166/scratchpad/current_episode.txt"
TIMING_LOG="/mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__sweep_6cam_newdemos_70/view_generation_times.txt"

# random float in [min, max], reseeded from bash's own $RANDOM each call
rand_float() {
  awk -v min="$1" -v max="$2" -v seed="$RANDOM$RANDOM$$" 'BEGIN{srand(seed); printf "%.4f", min + rand()*(max-min)}'
}

cd /home/puneeth/Desktop/VISTA

for EP_NUM in "$@"; do
  NOW_EPOCH=$(date +%s)
  if [ "$NOW_EPOCH" -ge "$CUTOFF_EPOCH" ]; then
    echo "=== CUTOFF REACHED ($(date)), STOPPING BEFORE STARTING episode_$EP_NUM ==="
    break
  fi

  EP="episode_$EP_NUM"
  BASE="$BASE_ROOT/$EP"
  if [ ! -d "$BASE/cam0" ]; then
    echo "=== SKIPPING $EP (no cam0 dir) ==="
    continue
  fi

  echo "$EP" > "$STATE_FILE"

  for VIEW in 1 2 3; do
    SEED=$RANDOM
    ANGLE_STD=$(rand_float 0.05 0.20)
    POS_STD=$(rand_float 0.02 0.10)
    OUT_DIR="$BASE/cam0_novel_view${VIEW}"
    echo "=== $EP VIEW${VIEW} START ($(date '+%H:%M:%S')) -- seed=$SEED angle_std=$ANGLE_STD position_std=$POS_STD ==="
    VIEW_START=$(date +%s)
    python vipl/vipl/scripts/generate_novel_view_video.py --input_dir "$BASE/cam0" --output_dir "$OUT_DIR" $COMMON --angle_std "$ANGLE_STD" --position_std "$POS_STD" --seed "$SEED"
    VIEW_END=$(date +%s)
    VIEW_SECS=$((VIEW_END - VIEW_START))
    printf "%s\t%ds\t%s\n" "$OUT_DIR" "$VIEW_SECS" "$(date -u -d @${VIEW_SECS} +%H:%M:%S)" >> "$TIMING_LOG"
    echo "=== $EP VIEW${VIEW} DONE (${VIEW_SECS}s) ==="
  done
  echo "=== $EP ALL COMPLETE ($(date '+%H:%M:%S')) ==="
done

echo "=== OVERNIGHT RUN FINISHED (all episodes done or cutoff reached) ==="
