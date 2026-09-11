# python run_vista_on_images.py   /mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__sort_100demos_6view/episode_000003/novel_views_newcam0   \
# --output_dir /mnt/cps_persistent1_shared/puneeth/experiments/results/temporal/pickplace/vista/ep56   \
# --model zeronvs_robot_6task_40k   --tx -0.4 --ty 0.0 --tz 0.0 --roll 5  --pitch -25 --yaw -5  --camera_convention opengl

# python run_vista_on_images.py   --input_dir /mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__pickplace_6cam_newdemos_20260526_161032/episode_000059/cam0   \
# --output_dir /mnt/cps_persistent1_shared/puneeth/experiments/results/temporal/pickplace/vista/ep59   \
# --model zeronvs_robot_6task_40k   --tx -0.4 --ty 0.0 --tz 0.0 --roll 5  --pitch -25 --yaw -5  --camera_convention opengl

# python run_vista_on_images.py   --input_dir /mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__pickplace_6cam_newdemos_20260526_161032/episode_000078/cam0   \
# --output_dir /mnt/cps_persistent1_shared/puneeth/experiments/results/temporal/pickplace/vista/ep78   \
# --model zeronvs_robot_6task_40k   --tx -0.4 --ty 0.0 --tz 0.0 --roll 5  --pitch -25 --yaw -5  --camera_convention opengl

# python run_vista_on_images.py   --input_dir /mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__pickplace_6cam_newdemos_20260526_161032/episode_000086/cam0   \
# --output_dir /mnt/cps_persistent1_shared/puneeth/experiments/results/temporal/pickplace/vista/ep86   \
# --model zeronvs_robot_6task_40k   --tx -0.4 --ty 0.0 --tz 0.0 --roll 5  --pitch -25 --yaw -5  --camera_convention opengl


python run_vista_on_images.py   --input_dir /mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__sort_100demos_6view/episode_000014/cam1   --output_dir /mnt/cps_persistent1_shared/puneeth/experiments/results/temporal/sort/vista/ep14   --model zeronvs_lpips_guard   --tx 0.5 --ty 0.0 --tz 0.0 --roll 5  --pitch 25 --yaw 10  --camera_convention opengl

python run_vista_on_images.py   --input_dir /mnt/cps_persistent1_shared/puneeth/experiments/saipuneethgottam__sort_100demos_6view/episode_000033/cam1   --output_dir /mnt/cps_persistent1_shared/puneeth/experiments/results/temporal/sort/vista/ep33   --model zeronvs_lpips_guard   --tx 0.5 --ty 0.0 --tz 0.0 --roll 5  --pitch 25 --yaw 10  --camera_convention opengl

