"""
Generate novel-view frames for an entire folder of video frames (all from one input angle),
rendering every frame from the same novel target camera angle, using the ZeroNVS diffusion model.

Example:
    python generate_novel_view_video.py \\
        --input_dir /path/to/frames --output_dir /path/to/novel_view_frames \\
        --checkpoint_path <ckpt> --config_path <config> \\
        --precomputed_scale 0.52 --guidance_scale 3.0 --ddim_steps 200 \\
        --angle_std 0.075 --position_std 0.03 --lpips_threshold 0.8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ZeroNVS/resources.py must take precedence over the same-named module shadowed
# by the zero123 editable install; only guaranteed by putting ZeroNVS root first.
sys.path.insert(0, "/home/puneeth/Desktop/ZeroNVS")

# checkpoints pickled under numpy>=2.0 reference numpy._core, which doesn't exist
# in the numpy 1.24 this env is pinned to (numpy 2.x isn't installable on Python
# 3.8 at all); alias the old module layout so torch.load can still unpickle them.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.umath", np.core.umath)
sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)

from vipl.models.augmentation import get_model_by_name
from vipl.utils.cam_utils import posori_to_rotmat
from vipl.utils.camera_pose_sampler import random_camera_perturbation
from vipl.utils.constants import ZERONVS_CONFIG_PATH

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder of input video frames (single-angle)")
    parser.add_argument("--output_dir", required=True, help="Folder to save the novel-view frames into (created if missing)")
    parser.add_argument(
        "--model_name",
        default="zeronvs_mimicgen_ft",
        choices=["zeronvs", "zeronvs_lpips_guard", "zeronvs_lpips_guard_real", "zeronvs_ft", "zeronvs_mimicgen_ft"],
        help="Which checkpoint/config from vipl.utils.constants to use. Ignored if --checkpoint_path is set.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to a custom .ckpt file to load instead of --model_name (e.g. a fine-tuning sweep checkpoint). "
             "Uses ZERONVS_CONFIG_PATH from vipl.utils.constants for the model config unless --config_path is set.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Model config yaml to pair with --checkpoint_path. Must match the config the checkpoint was actually "
             "trained with (e.g. conditioning_config.params.mode/embedding_dim, use_ema) or sampling will be "
             "garbage even though the checkpoint loads without error. Defaults to ZERONVS_CONFIG_PATH if not set.",
    )
    parser.add_argument("--angle_std", type=float, default=0.2, help="Std dev (radians) of the novel camera rotation, sampled once for the whole video")
    parser.add_argument("--position_std", type=float, default=0.05, help="Std dev (meters) of the novel camera translation, sampled once for the whole video")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling the (single, shared) novel camera pose")
    parser.add_argument(
        "--lpips_threshold",
        type=float,
        default=0.5,
        help="Only used with --checkpoint_path. Per-frame LPIPS guard: if a frame's generated view stays above "
             "this after all retries, that frame falls back to its original (unchanged) image. Since this is "
             "evaluated independently per frame, a low threshold can cause some frames in the sequence to "
             "silently stay unconverted while others succeed. Raise this (e.g. 0.9) for more consistent output "
             "across the whole video, at the cost of letting through lower-confidence generations.",
    )
    parser.add_argument(
        "--precomputed_scale",
        type=float,
        default=0.6,
        help="Only used with --checkpoint_path. Scene-scale value baked into the camera pose conditioning; must "
             "match (roughly) what the checkpoint was trained/calibrated with, not just be a fixed default.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Only used with --checkpoint_path. Classifier-free guidance scale for DDIM sampling.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=250,
        help="Only used with --checkpoint_path. Number of DDIM sampling steps.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not frame_paths:
        raise ValueError(f"No image files found in {input_dir}")
    print(f"Found {len(frame_paths)} frames in {input_dir}")

    # sample ONE novel target camera pose, shared across every frame, so the
    # whole video is rendered from the same consistent new viewpoint
    if args.seed is not None:
        np.random.seed(args.seed)
    identity_position = np.zeros(3)
    identity_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw quaternion, no rotation
    perturbation_quat = random_camera_perturbation(angle_std=args.angle_std)
    target_position = identity_position + np.random.normal(scale=args.position_std, size=3)
    target_orientation = perturbation_quat  # perturbation applied directly since starting orientation is identity

    original_camera = posori_to_rotmat(identity_position, identity_orientation)
    target_camera = posori_to_rotmat(target_position, target_orientation)

    camera_pose_path = output_dir / "camera_pose.json"
    with open(camera_pose_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "angle_std": args.angle_std,
                "position_std": args.position_std,
                "original_position": identity_position.tolist(),
                "original_orientation_xyzw": identity_orientation.tolist(),
                "target_position": target_position.tolist(),
                "target_orientation_xyzw": target_orientation.tolist(),
                "original_camera_cam2world": original_camera.tolist(),
                "target_camera_cam2world": target_camera.tolist(),
                "convention": "opengl",
            },
            f,
            indent=2,
        )
    print(f"Saved camera pose to {camera_pose_path}")

    if args.checkpoint_path is not None:
        from vipl.models.augmentation.zeronvs_aug import ZeroNVSModel
        print(f"Loading custom checkpoint '{args.checkpoint_path}'...")
        model = ZeroNVSModel(
            checkpoint=args.checkpoint_path,
            config=args.config_path or ZERONVS_CONFIG_PATH,
            zeronvs_params=dict(
                ddim_steps=args.ddim_steps,
                ddim_eta=1.0,
                precomputed_scale=args.precomputed_scale,
                guidance_scale=args.guidance_scale,
                lpips_loss_threshold=args.lpips_threshold,
            ),
        )
    else:
        print(f"Loading model '{args.model_name}'...")
        model = get_model_by_name(args.model_name)

    for i, frame_path in enumerate(frame_paths):
        print(f"[{i + 1}/{len(frame_paths)}] {frame_path.name}")
        original_image = Image.open(frame_path).convert("RGB")
        original_size = original_image.size  # augment() always works at a fixed 256x256 internally

        novel_view = model.augment(
            original_image=original_image,
            original_camera=original_camera,
            target_camera=target_camera,
            convention="opengl",  # posori_to_rotmat returns cam2world already in opengl convention
        )
        if novel_view.size != original_size:
            novel_view = novel_view.resize(original_size, resample=Image.LANCZOS)

        out_path = output_dir / frame_path.name
        novel_view.save(out_path)

    print(f"Done. Saved {len(frame_paths)} novel-view frames to {output_dir}")


if __name__ == "__main__":
    main()
