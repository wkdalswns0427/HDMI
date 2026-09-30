#!/usr/bin/env python3

import argparse
import numpy as np

from active_adaptation.utils.motion import MotionDataset


def main():
    parser = argparse.ArgumentParser(
        description="Check distance between two robot wrist links."
    )

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=True,
        help="Directory containing motion.npz and meta.json",
    )

    parser.add_argument(
        "--right",
        type=str,
        default="right_wrist_yaw_link",
    )

    parser.add_argument(
        "--left",
        type=str,
        default="left_wrist_yaw_link",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
    )

    args = parser.parse_args()

    dataset = MotionDataset.create_from_path(
        args.motion_dir,
        target_fps=args.fps,
    ).to("cpu")

    motion = dataset.data

    print("\n=== MOTION ===")
    print(f"path   : {args.motion_dir}")
    print(f"frames : {dataset.num_steps}")
    print(f"fps    : {args.fps}")
    print(f"bodies : {len(dataset.body_names)}")

    if args.right not in dataset.body_names:
        raise ValueError(
            f"{args.right} not found.\n"
            f"Available bodies:\n{dataset.body_names}"
        )

    if args.left not in dataset.body_names:
        raise ValueError(
            f"{args.left} not found.\n"
            f"Available bodies:\n{dataset.body_names}"
        )

    right_idx = dataset.body_names.index(args.right)
    left_idx = dataset.body_names.index(args.left)

    print("\n=== WRIST LINKS ===")
    print(f"right : {args.right} (index {right_idx})")
    print(f"left  : {args.left} (index {left_idx})")

    right_pos = motion.body_pos_w[:, right_idx].numpy()
    left_pos = motion.body_pos_w[:, left_idx].numpy()

    grip_span = np.linalg.norm(
        right_pos - left_pos,
        axis=1,
    )

    print("\n=== GRIP SPAN ===")
    print(f"mean   : {grip_span.mean():.4f} m")
    print(f"median : {np.median(grip_span):.4f} m")
    print(f"std    : {grip_span.std():.4f} m")
    print(f"min    : {grip_span.min():.4f} m")
    print(f"max    : {grip_span.max():.4f} m")
    print(f"range  : {np.ptp(grip_span):.4f} m")

    min_frame = int(np.argmin(grip_span))
    max_frame = int(np.argmax(grip_span))

    print("\n=== EXTREMES ===")
    print(
        f"min : frame {min_frame:4d} "
        f"({min_frame / args.fps:.2f} s) "
        f"= {grip_span[min_frame]:.4f} m"
    )

    print(
        f"max : frame {max_frame:4d} "
        f"({max_frame / args.fps:.2f} s) "
        f"= {grip_span[max_frame]:.4f} m"
    )

    step = max(1, int(args.fps))

    print("\n=== FRAME SAMPLES ===")

    for i in range(0, len(grip_span), step):
        print(
            f"frame {i:4d} | "
            f"{i / args.fps:6.2f} s | "
            f"{grip_span[i]:.4f} m"
        )


if __name__ == "__main__":
    main()