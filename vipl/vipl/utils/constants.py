ZERONVS_CONFIG_PATH = "/home/puneeth/Desktop/VISTA_Data/models/zeronvs_config.yaml"
ZERONVS_CHECKPOINT_PATH = "/home/puneeth/Desktop/VISTA_Data/models/zeronvs.ckpt"
ZERONVS_MIMICGEN_PATH = "/home/puneeth/Desktop/VISTA_Data/models/zeronvs_ft_mimicgen.ckpt"
ZERONVS_DROID_PATH = None # e.g. "abc/def/zeronvs_ft_droid.hdf5"

# mode=7dof, use_ema=false; the "-v1" checkpoint is the properly-saved step-40000
# snapshot (with optimizer state) -- last.ckpt in the same folder is a broken,
# weights-only save from later in the same run and produces pure noise.
ZERONVS_ROBOT_6TASK_CONFIG_PATH = "/home/puneeth/Desktop/VISTA_Data/models/zeronvs_config_robot_finetune.yaml"
ZERONVS_ROBOT_6TASK_40K_PATH = (
    "/mnt/cps_persistent1_shared/puneeth/zeronvs_robot_6task_finetune_logs/"
    "2026-05-13T20-42-00_sd-objaverse-finetune-c_concat-256/checkpoints/"
    "epoch=000003-step=000040000-v1.ckpt"
)