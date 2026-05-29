#!/usr/bin/env python3
"""Animal3D: render 8-view masks from OBJ meshes, then extract contours.

Animal3D project page: https://xujiacong.github.io/Animal3D/
Animal3D data: https://drive.google.com/drive/folders/17KRe8Z7jCZNDeBu45Wx2zS8Yh2tV_t2v

Main logic: render 8-view (8 azimuths of the same animal) binary masks for every OBJ under `obj_dir` (render_masks()), extract contours and preprocess(extract_contours()), save to `contours.pt` (torch tensor (N, N_POINTS, 2)).
Because rendering the masks takes a long time, you can also skip this step if you have the masks already (e.g. from rendered_masks_8views.zip (120MB)).
Otherwise, generating the masks requires `obj_files/{train,test}/{class}/*.obj` (310MB) downloaded from the Animal3D Google Drive.

By default, this script also does the following (see flags to turn off):
- Resamples each contour to 120 points (using a cubic spline and then uniform arc-length resampling)
- Centers each contour at the origin (subtract centroid)
- Rolls the contour so the start point is the point closest to the vertical drop from the centroid to the minimum y-coordinate
- Plots an example grid of 50 random contours for visualization (contour_grid.png)

Example usage, skipping rendering and using existing masks:
python extract_animal3d_contours.py \
  --skip-render \
  --masks-dir ./animal3d/rendered_masks_8views \
  -o ./animal3d/contours.pt

Dependencies: numpy, opencv-python, scikit-image, scipy, torch, pytorch3d, tqdm, matplotlib
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import interpolate as interp
from skimage import measure
from tqdm import tqdm

from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

# Camera azimuth angles (degrees) for each rendered view.
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
# PyTorch3D look_at_view_transform: distance from mesh origin to camera.
RENDER_DIST = 2.5
# Camera elevation above the horizontal plane (degrees).
RENDER_ELEV = 10
# Width and height of rendered silhouette PNGs (pixels).
RENDER_SIZE = 1024
# Soft-silhouette alpha threshold (0–255) for binarizing renders to white-on-black masks.
RENDER_THRESH = 127
# Default resampled contour length; minimum raw points required to resample.
DEFAULT_N_POINTS = 120
MIN_CONTOUR_PTS = 50


def collect_obj_files(obj_dir: str) -> list[tuple[str, str, str, str]]:
    """List (split, class, stem, path) for every `.obj` under `obj_dir`."""
    items = []
    for split in ("train", "test"):
        split_dir = osp.join(obj_dir, split)
        if not osp.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = osp.join(split_dir, cls)
            if not osp.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".obj"):
                    items.append((split, cls, fname[:-4], osp.join(cls_dir, fname)))
    return items


def _build_renderers(device: torch.device) -> list[MeshRenderer]:
    """Build one PyTorch3D silhouette renderer per azimuth in `AZIMUTHS`.

    Each renderer pairs a perspective camera (fixed distance, elevation, and azimuth) with a rasterizer and `SoftSilhouetteShader`. 
    The rasterizer projects the mesh into a square image and records which triangle faces hit each pixel; the shader turns those hits into a smooth alpha matte. 
    Renderers are built once and reused for every OBJ so all animals share the same eight camera poses.
    """
    raster = RasterizationSettings(
        image_size=RENDER_SIZE, blur_radius=1e-6, faces_per_pixel=50
    )
    blend = BlendParams(sigma=1e-4, gamma=1e-4)
    renderers = []
    for az in AZIMUTHS:
        r, t = look_at_view_transform(dist=RENDER_DIST, elev=RENDER_ELEV, azim=az)
        cam = FoVPerspectiveCameras(R=r, T=t, device=device)
        renderers.append(
            MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
                shader=SoftSilhouetteShader(blend_params=blend),
            )
        )
    return renderers


def _render_mesh(
    obj_path: str,
    out_paths: list[str],
    renderers: list[MeshRenderer],
    device: torch.device,
) -> int:
    """Render one mesh to 8 views; same logic as `render_animal3d_simple.render_mesh`."""
    verts, faces, _ = load_obj(obj_path, load_textures=False, device=device)
    verts[:, 1] *= -1
    verts = verts - verts.mean(0)
    verts = verts / verts.abs().max()
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
    saved = 0
    for renderer, out_path in zip(renderers, out_paths):
        alpha = renderer(mesh.extend(1))[0, ..., 3].cpu().numpy()
        mask = np.where(alpha > RENDER_THRESH / 255, 255, 0).astype(np.uint8)
        if mask.sum() == 0:
            continue
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, mask)
        saved += 1
    return saved


def render_masks(obj_dir: str, masks_dir: str, device: torch.device) -> None:
    """Render 8-view binary silhouettes for every OBJ under `obj_dir`."""
    print(f"Rendering on {device} (render_animal3d_simple.py settings)")
    renderers = _build_renderers(device)
    n_views = len(AZIMUTHS)

    items = collect_obj_files(obj_dir)
    if not items:
        raise FileNotFoundError(f"No .obj files under {obj_dir}")

    for split, cls, stem, obj_path in tqdm(items, desc="render"):
        out_paths = [
            osp.join(masks_dir, split, cls, f"{stem}_view{i:02d}.png")
            for i in range(n_views)
        ]
        if all(osp.exists(p) for p in out_paths):
            continue
        try:
            _render_mesh(obj_path, out_paths, renderers, device)
        except Exception as e:
            tqdm.write(f"  Error {stem}: {e}")


def collect_mask_paths(masks_dir: str) -> list[str]:
    """Collect paths to multi-view silhouette PNGs (`*_view*.png`)."""
    paths = []
    for root, _, files in os.walk(masks_dir):
        for f in sorted(files):
            if f.endswith(".png") and "_view" in f:
                paths.append(osp.join(root, f))
    return paths


def start_index_below_centroid(x: np.ndarray, y: np.ndarray) -> int:
    """Contour index for start: closest to bottom of vertical line from centroid to ymin.

    Drop a vertical line from the centroid (mean x, mean y) down to the shape's minimum y.
    Take the contour point closest to that bottom point `(com_x, y_min)`; ties go to lowest y.
    """
    com_x = float(x.mean())
    y_min = float(y.min())
    dist = np.hypot(x - com_x, y - y_min)
    return int(np.lexsort((y, dist))[0])


def resample_contour_xy(
    x: np.ndarray, y: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Periodic cubic spline + uniform arc-length resample (extract_8view_contours)."""
    n_dense = max(8 * n_points, 512)
    try:
        tck, _ = interp.splprep([x, y], s=0, per=True, k=3, quiet=True)
    except Exception:
        return None

    x_d, y_d = interp.splev(np.linspace(0, 1, n_dense, endpoint=False), tck)
    x_d = np.asarray(x_d, dtype=np.float64)
    y_d = np.asarray(y_d, dtype=np.float64)

    x_closed = np.concatenate([x_d, x_d[:1]])
    y_closed = np.concatenate([y_d, y_d[:1]])
    seg = np.sqrt(np.diff(x_closed) ** 2 + np.diff(y_closed) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total_len = float(cum[-1])
    if total_len <= 1e-12:
        return None

    s_target = np.linspace(0.0, total_len, n_points, endpoint=False)
    return np.interp(s_target, cum, x_closed), np.interp(s_target, cum, y_closed)


def extract_shape_from_mask(
    mask: np.ndarray,
    *,
    center: bool = True,
    resample: bool = True,
    n_points: int = DEFAULT_N_POINTS,
    roll_start: bool = True,
) -> np.ndarray | None:
    """Extract contour from mask; pipeline order: resample → center → roll."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    h = mask.shape[0]
    if cv2.moments((mask > 127).astype(np.uint8))["m00"] <= 0:
        return None

    contour_list = measure.find_contours(mask.astype(np.float32), level=127.5)
    if not contour_list:
        return None
    c = max(contour_list, key=len)
    if len(c) < MIN_CONTOUR_PTS:
        return None

    y = (h - 1 - c[:, 0]).astype(np.float64)
    x = c[:, 1].astype(np.float64)

    if resample:
        out = resample_contour_xy(x, y, n_points)
        if out is None:
            return None
        x, y = out

    if center:
        x -= x.mean()
        y -= y.mean()

    if roll_start:
        start = start_index_below_centroid(x, y)
        x = np.roll(x, -start)
        y = np.roll(y, -start)

    return np.stack([x, y], axis=1).astype(np.float32)


def extract_contours(
    masks_dir: str,
    *,
    center: bool = True,
    resample: bool = True,
    n_points: int = DEFAULT_N_POINTS,
    roll_start: bool = True,
) -> tuple[list[np.ndarray], list[str]]:
    """Extract contours from every mask PNG under `masks_dir`."""
    paths = collect_mask_paths(masks_dir)
    if not paths:
        raise FileNotFoundError(f"No mask PNGs under {masks_dir}")

    contours: list[np.ndarray] = []
    kept_paths: list[str] = []
    for p in tqdm(paths, desc="contours"):
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        c = extract_shape_from_mask(
            mask,
            center=center,
            resample=resample,
            n_points=n_points,
            roll_start=roll_start,
        )
        if c is not None:
            contours.append(c)
            kept_paths.append(p)
    return contours, kept_paths


def _contour_for_plot(contour: np.ndarray, centered: bool) -> np.ndarray:
    """Center for display only when contours were saved uncentered."""
    if centered:
        return contour
    c = contour.astype(np.float32)
    return c - c.mean(axis=0, keepdims=True)


def plot_contour_grid(
    contours: list[np.ndarray],
    out_path: str,
    *,
    centered: bool = True,
    n: int = 50,
    ncol: int = 10,
    nrow: int = 5,
    seed: int = 0,
    dpi: int = 120,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """Save a grid of random contours (style similar to gutils.vis.plot_shape_grid).

    Each panel: scattered contour points, gray dot at origin (centroid), black marker at start.
    """
    if not contours:
        return

    rng = np.random.default_rng(seed)
    n_show = min(n, len(contours))
    idx = rng.choice(len(contours), size=n_show, replace=False)

    tab = plt.get_cmap("tab10").colors
    color_line = tab[0]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes: list[plt.Axes] = []
    all_pts: list[np.ndarray] = []

    for plot_i, ci in enumerate(idx):
        shape = _contour_for_plot(contours[ci], centered)
        ax = fig.add_subplot(nrow, ncol, plot_i + 1)
        axes.append(ax)

        ax.scatter(shape[:, 0], shape[:, 1], s=4, color=color_line, zorder=3)
        ax.scatter([0.0], [0.0], s=12, color="gray", zorder=10)
        ax.scatter([shape[0, 0]], [shape[0, 1]], s=24, color="black", marker="o", zorder=11)
        ax.set_aspect("equal")
        ax.axis("off")
        all_pts.append(shape)

    for ax, shape in zip(axes, all_pts):
        x_min, x_max = shape[:, 0].min(), shape[:, 0].max()
        y_min, y_max = shape[:, 1].min(), shape[:, 1].max()
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        half = max(x_max - x_min, y_max - y_min) / 2
        half = half * 1.05 + 1e-6
        ax.set_xlim(x_c - half, x_c + half)
        ax.set_ylim(y_c - half, y_c + half)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.08, hspace=0.08)
    fig.suptitle(f"Random {n_show} contours (origin = centroid, dot = start)", fontsize=14)
    os.makedirs(osp.dirname(osp.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Render Animal3D masks and extract contours.")
    p.add_argument("--data-dir", default="./animal3d", help="Dataset root (contains obj_files/)")
    p.add_argument("--obj-dir", default=None, help="Override obj_files/ path")
    p.add_argument(
        "--masks-dir",
        default=None,
        help="Where to write/read mask PNGs (default: data-dir/rendered_masks_8views)",
    )
    p.add_argument("--skip-render", action="store_true", help="Only extract contours from existing masks")
    p.add_argument(
        "--device",
        default="cuda",
        help="torch device for rendering (e.g. cuda, cuda:0, cuda:1, cpu)",
    )
    p.add_argument("-o", "--output", default="contours.pt", help="Output .pt (list of variable-length contours)")
    p.add_argument(
        "--vis-out",
        default=None,
        help="PNG grid of 50 random contours (default: <output-stem>_grid.png)",
    )
    p.add_argument("--vis-seed", type=int, default=0, help="RNG seed for contour grid sample")
    p.add_argument("--center", action="store_true", default=True, help="Subtract contour centroid")
    p.add_argument("--no-center", dest="center", action="store_false")
    p.add_argument("--resample", action="store_true", default=True, help="Spline + equal arc-length resample, default true.")
    p.add_argument("--no-resample", dest="resample", action="store_false")
    p.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS, help="Points per contour when resampling")
    p.add_argument(
        "--roll-start",
        action="store_true",
        default=True,
        help="Start index = contour point closest to vertical drop from centroid to ymin",
    )
    p.add_argument("--no-roll-start", dest="roll_start", action="store_false")
    args = p.parse_args()

    data_dir = osp.abspath(args.data_dir)
    obj_dir = osp.abspath(args.obj_dir or osp.join(data_dir, "obj_files"))
    masks_dir = osp.abspath(args.masks_dir or osp.join(data_dir, "rendered_masks_8views"))

    if not args.skip_render:
        dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        render_masks(obj_dir, masks_dir, dev)

    contours, paths = extract_contours(
        masks_dir,
        center=args.center,
        resample=args.resample,
        n_points=args.n_points,
        roll_start=args.roll_start,
    )
    if not contours:
        raise RuntimeError("No contours extracted.")

    os.makedirs(osp.dirname(osp.abspath(args.output)) or ".", exist_ok=True)
    payload: dict = {"contours": contours, "paths": paths, "center": args.center, "resample": args.resample}
    if args.resample:
        payload["n_points"] = args.n_points
        payload["contours_stacked"] = torch.from_numpy(np.stack(contours, axis=0))
    torch.save(payload, args.output)
    shape_desc = f"({len(contours)}, {args.n_points}, 2)" if args.resample else f"{len(contours)} variable-length"
    print(f"Saved {shape_desc} → {args.output}")

    vis_out = args.vis_out or "contour_grid.png"
    plot_contour_grid(contours, vis_out, centered=args.center, seed=args.vis_seed)
    print(f"Saved contour grid for visualization → {vis_out}")


if __name__ == "__main__":
    main()
