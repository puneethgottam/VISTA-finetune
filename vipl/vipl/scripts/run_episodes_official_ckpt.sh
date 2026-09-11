#!/bin/bash
# Runs generate_novel_view_video.py (2 views) for a sequence of episodes using
# the ORIGINAL/official pretrained zeronvs_ft_mimicgen checkpoint (not the
# custom robot fine-tune), saving to separate "*_official" output folders.
set -e

BASE_ROOT="/mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__sweep_6cam_newdemos_70"
CKPT="/home/puneeth/Desktop/VISTA_Data/models/zeronvs_ft_mimicgen.ckpt"
CONFIG="/home/puneeth/Desktop/VISTA_Data/models/zeronvs_config.yaml"
COMMON="--checkpoint_path $CKPT --config_path $CONFIG --precomputed_scale 0.6 --guidance_scale 7.5 --ddim_steps 200 --lpips_threshold 0.8"
STATE_FILE="/tmp/claude-1006/-home-puneeth-Desktop-VISTA/d3505877-ed80-468e-9ecf-022087336166/scratchpad/current_episode_official.txt"
TIMING_LOG="/mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__sweep_6cam_newdemos_70/view_generation_times_official.txt"

# random float in [min, max], reseeded from bash's own $RANDOM each call
rand_float() {
  awk -v min="$1" -v max="$2" -v seed="$RANDOM$RANDOM$$" 'BEGIN{srand(seed); printf "%.4f", min + rand()*(max-min)}'
}

cd /home/puneeth/Desktop/VISTA

for EP_NUM in "$@"; do
  EP="episode_$EP_NUM"
  BASE="$BASE_ROOT/$EP"
  if [ ! -d "$BASE/cam0" ]; then
    echo "=== SKIPPING $EP (no cam0 dir) ==="
    continue
  fi

  echo "$EP" > "$STATE_FILE"

  for VIEW in 1 2; do
    SEED=$RANDOM
    ANGLE_STD=$(rand_float 0.05 0.20)
    POS_STD=$(rand_float 0.02 0.10)
    OUT_DIR="$BASE/cam0_novel_view${VIEW}_official"
    echo "=== $EP OFFICIAL_VIEW${VIEW} START ($(date '+%H:%M:%S')) -- seed=$SEED angle_std=$ANGLE_STD position_std=$POS_STD ==="
    VIEW_START=$(date +%s)
    python vipl/vipl/scripts/generate_novel_view_video.py --input_dir "$BASE/cam0" --output_dir "$OUT_DIR" $COMMON --angle_std "$ANGLE_STD" --position_std "$POS_STD" --seed "$SEED"
    VIEW_END=$(date +%s)
    VIEW_SECS=$((VIEW_END - VIEW_START))
    printf "%s\t%ds\t%s\n" "$OUT_DIR" "$VIEW_SECS" "$(date -u -d @${VIEW_SECS} +%H:%M:%S)" >> "$TIMING_LOG"
    echo "=== $EP OFFICIAL_VIEW${VIEW} DONE (${VIEW_SECS}s) ==="
  done
  echo "=== $EP OFFICIAL ALL COMPLETE ($(date '+%H:%M:%S')) ==="
done

echo "=== OFFICIAL CHECKPOINT RUN FINISHED ==="
