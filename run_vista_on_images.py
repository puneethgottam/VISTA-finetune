#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parent
ZERONVS_ROOT = REPO_ROOT.parent / "ZeroNVS"

for extra_path in [
    ZERONVS_ROOT / "zeronvs_diffusion" / "zero123",
    ZERONVS_ROOT / "threestudio",
]:
    extra_path_str = str(extra_path)
    if extra_path.exists() and extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from vipl.models.augmentation import get_model_by_name
from vipl.utils.constants import (
    ZERONVS_CHECKPOINT_PATH,
    ZERONVS_CONFIG_PATH,
    ZERONVS_DROID_PATH,
    ZERONVS_MIMICGEN_PATH,
    ZERONVS_ROBOT_6TASK_40K_PATH,
)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def matrix_from_value(value, field_name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"{field_name} must be a 4x4 matrix, got shape {arr.shape}")
    return arr


def make_cam2world(tx=0.0, ty=0.0, tz=0.0, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()
    cam2world[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return cam2world


def camera_json_to_cam2world(camera):
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = np.asarray(camera["rotation"], dtype=np.float32)
    cam2world[:3, 3] = np.asarray(camera["position"], dtype=np.float32)
    return cam2world


def load_robot_camera_poses(cameras_json, source_camera_id, target_camera_id):
    cameras = {int(camera["id"]): camera for camera in load_json(cameras_json)}
    missing = [
        camera_id
        for camera_id in (source_camera_id, target_camera_id)
        if camera_id not in cameras
    ]
    if missing:
        raise ValueError(f"Missing camera ids in {cameras_json}: {missing}")
    return (
        camera_json_to_cam2world(cameras[source_camera_id]),
        camera_json_to_cam2world(cameras[target_camera_id]),
    )


def sample_random_pose(rng, translation_std, rotation_std_deg):
    translation = rng.normal(loc=0.0, scale=translation_std, size=3).astype(np.float32)
    rotation = rng.normal(loc=0.0, scale=rotation_std_deg, size=3).astype(np.float32)
    return make_cam2world(
        tx=float(translation[0]),
        ty=float(translation[1]),
        tz=float(translation[2]),
        roll_deg=float(rotation[0]),
        pitch_deg=float(rotation[1]),
        yaw_deg=float(rotation[2]),
    )


def resolve_image_paths(input_dir, explicit_images=None):
    input_dir = Path(input_dir)
    if explicit_images:
        return [input_dir / image_name for image_name in explicit_images]
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def build_jobs(spec, image_paths):
    shared_original = spec.get("original_camera")
    shared_targets = spec.get("target_cameras", {})
    per_image = spec.get("images", {})
    jobs = []

    for image_path in image_paths:
        image_name = image_path.name
        image_spec = per_image.get(image_name, {})

        original_camera = image_spec.get("original_camera", shared_original)
        target_camera = image_spec.get("target_camera", shared_targets.get(image_name))

        if original_camera is None:
            raise ValueError(
                f"Missing original_camera for {image_name}. "
                "Provide a top-level original_camera or an images.<name>.original_camera."
            )
        if target_camera is None:
            raise ValueError(
                f"Missing target_camera for {image_name}. "
                "Provide images.<name>.target_camera or target_cameras.<name>."
            )

        jobs.append(
            {
                "image_path": image_path,
                "original_camera": matrix_from_value(original_camera, f"{image_name}.original_camera"),
                "target_camera": matrix_from_value(target_camera, f"{image_name}.target_camera"),
                "output_name": image_spec.get("output_name", image_path.stem + "_vista.png"),
            }
        )
    return jobs


def build_jobs_from_args(args, image_paths):
    rng = np.random.default_rng(args.seed)
    original_camera = make_cam2world(
        tx=args.source_tx,
        ty=args.source_ty,
        tz=args.source_tz,
        roll_deg=args.source_roll,
        pitch_deg=args.source_pitch,
        yaw_deg=args.source_yaw,
    )
    jobs = []
    for image_path in image_paths:
        if args.random_target_pose:
            target_camera = sample_random_pose(
                rng=rng,
                translation_std=args.random_translation_std,
                rotation_std_deg=args.random_rotation_std_deg,
            )
        else:
            target_camera = make_cam2world(
                tx=args.tx,
                ty=args.ty,
                tz=args.tz,
                roll_deg=args.roll,
                pitch_deg=args.pitch,
                yaw_deg=args.yaw,
            )
        jobs.append(
            {
                "image_path": image_path,
                "original_camera": original_camera.copy(),
                "target_camera": target_camera,
                "output_name": image_path.stem + "_vista.png",
            }
        )
    return jobs


def build_jobs_from_robot_cameras(args, image_paths):
    original_camera, target_camera = load_robot_camera_poses(
        args.cameras_json,
        args.source_camera_id,
        args.target_camera_id,
    )
    return [
        {
            "image_path": image_path,
            "original_camera": original_camera.copy(),
            "target_camera": target_camera.copy(),
            "output_name": image_path.stem + f"_cam{args.target_camera_id}.png",
        }
        for image_path in image_paths
    ]


def build_jobs_from_relative_source_camera(args, image_paths):
    cameras = {int(camera["id"]): camera for camera in load_json(args.cameras_json)}
    if args.source_camera_id not in cameras:
        raise ValueError(f"Missing source camera id {args.source_camera_id} in {args.cameras_json}")

    original_camera = camera_json_to_cam2world(cameras[args.source_camera_id])
    relative_target = make_cam2world(
        tx=args.tx,
        ty=args.ty,
        tz=args.tz,
        roll_deg=args.roll,
        pitch_deg=args.pitch,
        yaw_deg=args.yaw,
    )
    target_camera = original_camera @ relative_target
    pose_tag = (
        f"rel_tx{args.tx:g}_ty{args.ty:g}_tz{args.tz:g}_"
        f"r{args.roll:g}_p{args.pitch:g}_y{args.yaw:g}"
    ).replace("-", "m").replace(".", "p")

    return [
        {
            "image_path": image_path,
            "original_camera": original_camera.copy(),
            "target_camera": target_camera.copy(),
            "output_name": image_path.stem + f"_{pose_tag}.png",
        }
        for image_path in image_paths
    ]


def model_paths_for_config(model_name):
    checkpoint_by_model = {
        "zeronvs": ZERONVS_CHECKPOINT_PATH,
        "zeronvs_lpips_guard": ZERONVS_CHECKPOINT_PATH,
        "zeronvs_lpips_guard_real": ZERONVS_CHECKPOINT_PATH,
        "zeronvs_ft": ZERONVS_DROID_PATH,
        "zeronvs_mimicgen_ft": ZERONVS_MIMICGEN_PATH,
        "zeronvs_robot_6task_40k": ZERONVS_ROBOT_6TASK_40K_PATH,
    }
    checkpoint_path = checkpoint_by_model.get(model_name)
    if checkpoint_path is None:
        return {}
    return {
        "zeronvs_config_path": ZERONVS_CONFIG_PATH,
        "zeronvs_checkpoint_path": checkpoint_path,
    }


def save_run_config(output_dir, args, jobs):
    config = {
        "command_args": vars(args),
        "model_paths": model_paths_for_config(args.model),
        "images": [
            {
                "input_path": str(job["image_path"]),
                "output_name": job["output_name"],
                "original_camera": job["original_camera"].tolist(),
                "target_camera": job["target_camera"].tolist(),
            }
            for job in jobs
        ],
    }
    with open(output_dir / "vista_run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VISTA / ZeroNVS augmentation on a folder of images using camera poses from JSON or simple CLI pose offsets."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing input images.")
    parser.add_argument("--poses_json", default=None, help="Optional JSON file containing source / target camera matrices.")
    parser.add_argument("--output_dir", required=True, help="Directory to save augmented images.")
    parser.add_argument(
        "--cameras_json",
        default="/mnt/cps_scratch1_tmp/cameras.json",
        help="Robot camera calibration JSON used with --target_camera_id.",
    )
    parser.add_argument(
        "--model",
        default="zeronvs_mimicgen_ft",
        choices=[
            "zeronvs",
            "zeronvs_lpips_guard",
            "zeronvs_lpips_guard_real",
            "zeronvs_ft",
            "zeronvs_mimicgen_ft",
            "zeronvs_robot_6task_40k",
            "clone",
        ],
        help="VISTA augmentation model to use.",
    )
    parser.add_argument(
        "--camera_convention",
        default="opencv",
        choices=["opencv", "opengl"],
        help="Convention used by the 4x4 cam2world matrices in poses_json.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional pre-resize for input images before augmentation.",
    )
    parser.add_argument("--tx", type=float, default=0.0, help="Target camera x translation.")
    parser.add_argument("--ty", type=float, default=0.0, help="Target camera y translation.")
    parser.add_argument("--tz", type=float, default=0.0, help="Target camera z translation.")
    parser.add_argument("--roll", type=float, default=0.0, help="Target camera roll in degrees.")
    parser.add_argument("--pitch", type=float, default=0.0, help="Target camera pitch in degrees.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Target camera yaw in degrees.")
    parser.add_argument("--source_tx", type=float, default=0.0, help="Source camera x translation when not using poses_json.")
    parser.add_argument("--source_ty", type=float, default=0.0, help="Source camera y translation when not using poses_json.")
    parser.add_argument("--source_tz", type=float, default=0.0, help="Source camera z translation when not using poses_json.")
    parser.add_argument("--source_roll", type=float, default=0.0, help="Source camera roll in degrees when not using poses_json.")
    parser.add_argument("--source_pitch", type=float, default=0.0, help="Source camera pitch in degrees when not using poses_json.")
    parser.add_argument("--source_yaw", type=float, default=0.0, help="Source camera yaw in degrees when not using poses_json.")
    parser.add_argument(
        "--source_camera_id",
        type=int,
        default=0,
        help="Robot source camera id for calibration-based novel views.",
    )
    parser.add_argument(
        "--target_camera_id",
        type=int,
        default=None,
        help="Robot target camera id. If set, camera poses are read from --cameras_json and --tx/--pitch/etc are ignored.",
    )
    parser.add_argument(
        "--relative_to_source_camera",
        action="store_true",
        help=(
            "Use --source_camera_id as the input camera pose and apply "
            "--tx/--ty/--tz/--roll/--pitch/--yaw as a relative transform to make "
            "an arbitrary target camera pose."
        ),
    )
    parser.add_argument(
        "--random_target_pose",
        action="store_true",
        help="Sample a random target pose per image instead of using the fixed target pose offsets.",
    )
    parser.add_argument(
        "--random_translation_std",
        type=float,
        default=0.05,
        help="Translation stddev used by --random_target_pose.",
    )
    parser.add_argument(
        "--random_rotation_std_deg",
        type=float,
        default=0.0,
        help="Rotation stddev in degrees used by --random_target_pose.",
    )
    parser.add_argument(
        "--save_augmented_on_lpips_fail",
        action="store_true",
        help="Save the best generated view even when all attempts fail the LPIPS guard.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by --random_target_pose.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_json(args.poses_json) if args.poses_json else None
    image_paths = resolve_image_paths(input_dir, explicit_images=spec.get("image_order") if spec else None)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    if spec:
        jobs = build_jobs(spec, image_paths)
    elif args.target_camera_id is not None:
        jobs = build_jobs_from_robot_cameras(args, image_paths)
    elif args.relative_to_source_camera:
        jobs = build_jobs_from_relative_source_camera(args, image_paths)
    else:
        jobs = build_jobs_from_args(args, image_paths)
    save_run_config(output_dir, args, jobs)
    print('Loadig model', args.model)
    model = get_model_by_name(
        args.model,
        return_best_on_lpips_fail=args.save_augmented_on_lpips_fail,
    )

    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] Processing {job['image_path'].name}")
        image = Image.open(job["image_path"]).convert("RGB")
        target_size = image.size  # augment() always works at a fixed 256x256 internally
        if args.resize is not None:
            image = image.resize(tuple(args.resize), resample=Image.LANCZOS)

        out = model.augment(
            original_image=image,
            original_camera=job["original_camera"],
            target_camera=job["target_camera"],
            convention=args.camera_convention,
        )
        if out.size != target_size:
            out = out.resize(target_size, resample=Image.LANCZOS)
        out.save(output_dir / job["output_name"])

    print(f"Saved {len(jobs)} augmented image(s) to {output_dir}")


if __name__ == "__main__":
    main()
