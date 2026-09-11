"""
Generate a novel view of a single input image using the ZeroNVS diffusion model.

Example:
    python generate_novel_view.py --image_path motorcycle.png --output_path novel_view.png \
        --model_name zeronvs_mimicgen_ft --angle_std 0.3 --position_std 0.1
"""
import argparse
import sys

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
from vipl.utils.camera_pose_sampler import CameraPoseSampler, random_camera_perturbation
from vipl.utils.constants import ZERONVS_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", default="novel_view.png", help="Where to save the generated novel view")
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
             "trained with (e.g. conditioning_config.params.mode/embedding_dim) or sampling will be garbage even "
             "though the checkpoint loads without error. Defaults to ZERONVS_CONFIG_PATH if not set.",
    )
    parser.add_argument("--angle_std", type=float, default=0.2, help="Std dev (radians) of random camera rotation")
    parser.add_argument("--position_std", type=float, default=0.05, help="Std dev (meters) of random camera translation")
    parser.add_argument(
        "--lpips_threshold",
        type=float,
        default=0.5,
        help="Only used with --checkpoint_path. If the LPIPS distance between input and generated image stays "
             "above this after all retries, the original image is returned instead. Raise this (e.g. 0.9) to "
             "see low-guard-confidence generations instead of silently falling back to the input image.",
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

    original_image = Image.open(args.image_path).convert("RGB")
    original_size = original_image.size  # augment() always works at a fixed 256x256 internally

    # treat the input image's camera as the identity pose, and sample a
    # nearby target camera pose to render the scene from
    identity_position = np.zeros(3)
    identity_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw quaternion, no rotation
    perturbation_quat = random_camera_perturbation(angle_std=args.angle_std)
    target_position = identity_position + np.random.normal(scale=args.position_std, size=3)
    target_orientation = perturbation_quat  # perturbation applied directly since starting orientation is identity

    original_camera = posori_to_rotmat(identity_position, identity_orientation)
    target_camera = posori_to_rotmat(target_position, target_orientation)

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

    print("Generating novel view...")
    novel_view = model.augment(
        original_image=original_image,
        original_camera=original_camera,
        target_camera=target_camera,
        convention="opengl",  # posori_to_rotmat returns cam2world already in opengl convention
    )
    if novel_view.size != original_size:
        novel_view = novel_view.resize(original_size, resample=Image.LANCZOS)

    novel_view.save(args.output_path)
    print(f"Saved novel view to {args.output_path}")


if __name__ == "__main__":
    main()
