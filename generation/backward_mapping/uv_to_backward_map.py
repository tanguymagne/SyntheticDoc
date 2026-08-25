"""Invert the rendered UV maps into the backward maps used as unwarping supervision."""

import argparse
from pathlib import Path

import numpy as np
import OpenEXR
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from tqdm import tqdm

DEFAULT_PATTERN = "uv_inverse.exr"
DEFAULT_SIZE = (177, 121)

# The four directions a missing pixel can be extrapolated from, as (row, column) steps.
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def extrapolate(array, rows_nan, cols_nan, drow, dcol):
    """Extrapolate `array` at the given pixels from their two neighbours along (drow, dcol)."""
    h, w = array.shape
    rows1, cols1 = rows_nan + drow, cols_nan + dcol
    rows2, cols2 = rows_nan + 2 * drow, cols_nan + 2 * dcol

    # Pixels whose neighbours fall outside the array, or are missing themselves, extrapolate to NaN.
    inside = (rows2 >= 0) & (rows2 < h) & (cols2 >= 0) & (cols2 < w)
    rows1, cols1 = np.clip(rows1, 0, h - 1), np.clip(cols1, 0, w - 1)
    rows2, cols2 = np.clip(rows2, 0, h - 1), np.clip(cols2, 0, w - 1)

    # Constant slope from the second neighbour to the first one, continued to the target pixel.
    return np.where(inside, 2 * array[rows1, cols1] - array[rows2, cols2], np.nan)


def fill_missing(array):
    """Fill the NaNs of a 2D array by repeatedly extrapolating from its valid pixels."""
    filled = np.copy(array)
    rows_nan, cols_nan = np.where(np.isnan(filled))

    while rows_nan.size > 0:
        candidates = np.stack([extrapolate(filled, rows_nan, cols_nan, *d) for d in DIRECTIONS])

        # A single direction is too unreliable to extrapolate from, so two are required.
        fillable = np.count_nonzero(~np.isnan(candidates), axis=0) >= 2
        if not fillable.any():
            break

        filled[rows_nan[fillable], cols_nan[fillable]] = np.nanmean(candidates[:, fillable], axis=0)
        rows_nan, cols_nan = rows_nan[~fillable], cols_nan[~fillable]

    return filled


def uv_to_backward_map(uv_map, size=DEFAULT_SIZE):
    """Turn a rendered UV map into a backward map of `size`, in pixels of the rendered image."""
    height, width = size

    u_map = uv_map[:, :, 0]
    v_map = uv_map[:, :, 1]
    visible = uv_map[:, :, 2] > 0

    # Extract visible pixel coordinates
    us, vs = u_map[visible] * width, v_map[visible] * height
    rows, cols = np.where(visible)

    # Create regular grid of target coordinates to interpolate onto
    grid_u, grid_v = np.meshgrid(np.arange(width), np.arange(height))
    queries = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)

    # Build Delaunay triangulation shifted by half a pixel, to place grid samples at the centre of
    # each backward map pixel.
    triangulation = Delaunay(np.stack([us, vs], axis=1) - 0.5)

    # Interpolate the warped pixel coordinates at each unwarped pixel, extrapolate missing values.
    bm_x = LinearNDInterpolator(triangulation, cols, fill_value=np.nan)(queries).reshape(size)
    bm_y = LinearNDInterpolator(triangulation, rows, fill_value=np.nan)(queries).reshape(size)
    bm = np.dstack([fill_missing(bm_x), fill_missing(bm_y)])

    # V grows upwards in the UV map, whereas rows grow downwards in the backward map.
    return bm[::-1]


def save_backward_map(exr_path, size=DEFAULT_SIZE):
    """Convert the UV map of one EXR file into a backward map and save it as a .npy file."""
    uv_map = OpenEXR.File(str(exr_path)).parts[0].channels["RGB"].pixels

    bm = uv_to_backward_map(uv_map, size)

    # Normalized to [-1, 1], the range expected by grid sampling.
    bm = (bm / np.array([uv_map.shape[1], uv_map.shape[0]])) * 2 - 1

    np.save(exr_path.parent / "backward_map.npy", bm.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="EXR file holding a UV map, or a directory to search for them",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help=f"Name of the UV maps looked for in a directory. Default '{DEFAULT_PATTERN}'",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=DEFAULT_SIZE,
        help=f"Size of the backward map. Default {DEFAULT_SIZE[0]} {DEFAULT_SIZE[1]}",
    )
    args = parser.parse_args()

    if args.input.is_dir():
        exr_paths = sorted(args.input.rglob(args.pattern))
    else:
        exr_paths = [args.input]

    failed = 0
    for exr_path in tqdm(exr_paths):
        # A UV map with no visible page, left by a render that failed, has nothing to invert and
        # raises. It should not bring down a run over a whole directory.
        try:
            save_backward_map(exr_path, tuple(args.size))
        except Exception as error:
            tqdm.write(f"Skipped {exr_path}: {error}")
            failed += 1

    if failed:
        print(f"Skipped {failed} of the {len(exr_paths)} UV maps found.")
