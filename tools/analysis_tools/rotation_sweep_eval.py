#!/usr/bin/env python3
"""
Rotation robustness sweep:
Evaluate mAP50 and mAP(0.5~0.95) on the test set rotated by a fixed angle.

Strategy:
- Rotate images on-the-fly in the test pipeline via `FixedRotate` (config uses env ROT_ANGLE).
- Rotate GT polygons by the same angle (in the padded image coordinate system)
  and point `data.test.ann_file` to the rotated label folder.

This avoids copying/rotating the large .npy images on disk.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _infer_hw_from_any_npy(images_dir: Path) -> tuple[int, int]:
    # Images are stored as .npy, typically shape (C, W, H) for this project.
    p = next(images_dir.glob("*.npy"))
    arr = np.load(p)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected npy ndim for {p}: {arr.ndim}")
    # Heuristic: channels-first
    if arr.shape[0] <= 32 and arr.shape[1] > 32 and arr.shape[2] > 32:
        c, w, h = arr.shape
        return h, w  # (H, W)
    # channels-last
    if arr.shape[2] <= 32 and arr.shape[0] > 32 and arr.shape[1] > 32:
        h, w, c = arr.shape
        return h, w
    # Fallback: assume (C, H, W)
    if arr.shape[0] <= 32:
        c, h, w = arr.shape
        return h, w
    raise RuntimeError(f"Cannot infer H/W from shape {arr.shape} for {p}")


def _rotation_matrix_for_coords(h: int, w: int, angle_deg: float) -> np.ndarray:
    # Match PolyRandomRotate rm_coords (offset=0).
    center = (w / 2.0, h / 2.0)
    return cv2.getRotationMatrix2D(center, angle_deg, 1.0)


def _rotate_poly_xy(poly_xy: np.ndarray, rm: np.ndarray) -> np.ndarray:
    # poly_xy: (4,2)
    poly_xy = poly_xy.astype(np.float32, copy=False)
    out = cv2.transform(poly_xy[None, :, :], rm)[0]
    return out


def _rotate_label_file(src: Path, dst: Path, rm: np.ndarray) -> None:
    out_lines: list[str] = []
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                out_lines.append("")
                continue
            parts = s.split()
            # Keep metadata lines if present (some DOTA-style labels include gsd:...).
            if parts and parts[0].startswith("gsd"):
                out_lines.append(s)
                continue
            if len(parts) < 9:
                # Unknown/invalid line; keep as-is to avoid dropping info.
                out_lines.append(s)
                continue

            try:
                coords = np.array([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)
            except ValueError:
                out_lines.append(s)
                continue

            cls_name = parts[8]
            difficulty = parts[9] if len(parts) >= 10 else "0"

            coords_r = _rotate_poly_xy(coords, rm).reshape(-1)
            # Keep enough precision to avoid rounding artifacts affecting IoU/mAP.
            coord_str = " ".join(f"{v:.3f}" for v in coords_r.tolist())
            out_lines.append(f"{coord_str} {cls_name} {difficulty}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _pad_to_multiple(h: int, w: int, divisor: int) -> tuple[int, int]:
    if divisor <= 0:
        raise ValueError(f"pad divisor must be > 0, got {divisor}")
    pad_h = int(np.ceil(h / divisor) * divisor)
    pad_w = int(np.ceil(w / divisor) * divisor)
    return pad_h, pad_w


def ensure_rotated_labels(
    angle_deg: int,
    src_labels_dir: Path,
    dst_labels_dir: Path,
    h_pad: int,
    w_pad: int,
) -> None:
    # Use a marker file to allow resume without scanning all txts.
    marker = dst_labels_dir / ".done"
    if marker.exists():
        return

    rm = _rotation_matrix_for_coords(h_pad, w_pad, float(angle_deg))
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    for src in src_labels_dir.glob("*.txt"):
        dst = dst_labels_dir / src.name
        _rotate_label_file(src, dst, rm)
    marker.write_text(f"angle={angle_deg}\n", encoding="utf-8")


# Match rows from the ASCII table printed by mmrotate's eval_map logger.
_RE_MAP50 = re.compile(r"\|\s*mAP50\s*\|.*\|\s*([0-9.]+)\s*\|\s*$")
_RE_MAP5095 = re.compile(r"\|\s*mAP\(0\.5~0\.95\)\s*\|.*\|\s*([0-9.]+)\s*\|\s*$")


def parse_metrics_from_log(log_path: Path) -> tuple[float | None, float | None]:
    map50 = None
    map5095 = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _RE_MAP50.search(line)
        if m:
            map50 = float(m.group(1))
        m = _RE_MAP5095.search(line)
        if m:
            map5095 = float(m.group(1))
    return map50, map5095


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmrotate-root", type=Path, default=Path.cwd(), help="Path to mmrotate repo root")
    ap.add_argument("--python", type=Path, required=True, help="Python interpreter (e.g. conda env python)")
    ap.add_argument("--config", type=str, default="configs/oriented_reppoints/start_level_0_rotate_eval.py")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--angles", type=str, default="195:210:15", help="start:stop:step (stop exclusive), e.g. 0:360:15")
    ap.add_argument("--out-dir", type=Path, default=Path("/data1/users/hanshuaihao01/mmrotate/tools/analysis_tools"))
    ap.add_argument("--samples-per-gpu", type=int, default=1)
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--pad-divisor", type=int, default=32, help="Pad divisor used in the test pipeline")
    args = ap.parse_args()

    mmroot: Path = args.mmrotate_root.resolve()
    out_dir: Path = (mmroot / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = mmroot / "data/mod/test/images"
    src_labels_dir = mmroot / "data/mod/test/labels"
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images dir: {images_dir}")
    if not src_labels_dir.is_dir():
        raise SystemExit(f"Missing labels dir: {src_labels_dir}")

    h, w = _infer_hw_from_any_npy(images_dir)
    pad_h, pad_w = _pad_to_multiple(h, w, args.pad_divisor)

    start_s, stop_s, step_s = args.angles.split(":")
    start, stop, step = int(start_s), int(stop_s), int(step_s)
    angles = list(range(start, stop, step))

    csv_path = out_dir / "rotation_metrics.csv"
    rows = []
    if csv_path.exists():
        # Resume: load existing results.
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    done_angles = {int(r["angle_deg"]) for r in rows if "angle_deg" in r}

    for angle in angles:
        if angle in done_angles:
            continue

        angle_dir = out_dir / f"angle_{angle:03d}"
        labels_dir = angle_dir / "labels"
        work_dir = angle_dir / "work_dir"
        log_path = angle_dir / "test.log"

        ensure_rotated_labels(angle, src_labels_dir, labels_dir, h_pad=pad_h, w_pad=pad_w)

        env = os.environ.copy()
        env["ROT_ANGLE"] = str(angle)

        cmd = [
            str(args.python),
            "tools/test.py",
            args.config,
            args.checkpoint,
            "--eval",
            "mAP",
            "--work-dir",
            str(work_dir),
            "--cfg-options",
            f"data.test.ann_file={labels_dir.as_posix()}/",
            f"data.samples_per_gpu={args.samples_per_gpu}",
            f"data.workers_per_gpu={args.workers_per_gpu}",
        ]

        with log_path.open("w", encoding="utf-8") as log_f:
            p = subprocess.run(cmd, cwd=mmroot, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            rows.append(
                {
                    "angle_deg": str(angle),
                    "mAP50": "",
                    "mAP_0.5_0.95": "",
                    "status": f"error_exit_{p.returncode}",
                    "log": str(log_path),
                }
            )
        else:
            map50, map5095 = parse_metrics_from_log(log_path)
            rows.append(
                {
                    "angle_deg": str(angle),
                    "mAP50": "" if map50 is None else f"{map50:.3f}",
                    "mAP_0.5_0.95": "" if map5095 is None else f"{map5095:.3f}",
                    "status": "ok" if map50 is not None and map5095 is not None else "parsed_incomplete",
                    "log": str(log_path),
                }
            )

        # Persist after each angle (so we can stop/restart anytime).
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["angle_deg", "mAP50", "mAP_0.5_0.95", "status", "log"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: int(r["angle_deg"])))

    # Also write a simple markdown table.
    md_path = out_dir / "rotation_metrics.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| angle_deg | mAP50 | mAP(0.5~0.95) | status |\n")
        f.write("|---:|---:|---:|---|\n")
        for r in sorted(rows, key=lambda r: int(r["angle_deg"])):
            f.write(f"| {r['angle_deg']} | {r['mAP50']} | {r['mAP_0.5_0.95']} | {r['status']} |\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
