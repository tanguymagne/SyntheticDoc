import random
from pathlib import Path

import numpy as np

from .base_config_generator import ConfigGenerator


def load_obj_vertices(path):
    """Return the vertices of an .obj file as an (n, 3) array, in file order."""
    vertices = []
    with Path(path).open("r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)


def generate_roller_motion(dir_xy, center, start_offset, end_offset, z, t_start, t_end):
    """Build the motion rolling the cylinder over the sheet along `dir_xy`, to press the fold."""
    dir_xy = dir_xy / np.linalg.norm(dir_xy)

    start_xy = center[:2] + dir_xy * start_offset
    end_xy = center[:2] - dir_xy * end_offset

    # Angle so roller axis is perpendicular to motion
    angle = np.degrees(np.arctan2(dir_xy[1], dir_xy[0])) + 90

    motion = [
        {
            "time": t_start,
            "transform": {
                "translate": [float(start_xy[0]), float(start_xy[1]), z],
                "rotate": [angle, 0, 0, 1],
            },
        },
        {
            "time": t_end,
            "transform": {
                "translate": [float(end_xy[0]), float(end_xy[1]), z],
                "rotate": [angle, 0, 0, 1],
            },
        },
    ]

    return motion


def generate_fold_motion(v, center):
    """Build the motion folding point `v` over the sheet and back, and the direction it travels."""
    # Direction toward opposite side (through paper center)
    dir_xy = center[:2] - v[:2]
    dir_norm = np.linalg.norm(dir_xy)
    if dir_norm < 1e-6:
        dir_xy = np.random.randn(2)
        dir_norm = np.linalg.norm(dir_xy)
    dir_xy /= dir_norm

    # Travel distance
    distance = dir_norm * 2.0
    dist1 = distance * 0.5
    dist2 = distance * 1.0

    # Height settings
    up_rel = random.uniform(0.05, 0.08)
    lift_return_rel = 0.16
    base_offset_z = 0.002

    start_offset_xy = dir_xy * 0.005

    def make_xyz(dx, dy, dz_rel):
        """Turn an offset relative to the start position into an absolute translation."""
        x = float(start_offset_xy[0] + dx)
        y = float(start_offset_xy[1] + dy)
        z = base_offset_z + max(dz_rel, 0.0)  # disallow negative
        return [x, y, z]

    motion = [
        # 0: Start (slightly offset)
        {"time": 0, "transform": {"translate": make_xyz(0, 0, 0)}},
        # 1: Pull halfway up
        {
            "time": 1,
            "transform": {"translate": make_xyz(dir_xy[0] * dist1, dir_xy[1] * dist1, up_rel)},
        },
        # 2: Pull fully across and press down a bit
        {
            "time": 2,
            "transform": {"translate": make_xyz(dir_xy[0] * dist2, dir_xy[1] * dist2, 0.02)},
        },
        # 3: Hold pressed position (roller folds)
        {
            "time": 6,
            "transform": {"translate": make_xyz(dir_xy[0] * dist2, dir_xy[1] * dist2, 0.02)},
        },
        # 4: Lift up before returning — unfold motion
        {
            "time": 7,
            "transform": {
                "translate": make_xyz(
                    dir_xy[0] * dist2 * 0.7, dir_xy[1] * dist2 * 0.7, lift_return_rel
                )
            },
        },
        # 5: Midway return, still slightly lifted
        {
            "time": 8,
            "transform": {
                "translate": make_xyz(
                    dir_xy[0] * dist1 * 0.3, dir_xy[1] * dist1 * 0.3, lift_return_rel * 0.5
                )
            },
        },
        # 6: Back to origin, drop down gently
        {"time": 9, "transform": {"translate": make_xyz(0, 0, 0)}},
    ]

    return motion, dir_xy


class FoldByPullConfigGenerator(ConfigGenerator):
    """Fold a random point of the sheet outline over the sheet, press it with a roller, unfold."""

    def set_configs(self):
        """Load the fold base config and the outline vertices the handle is picked from."""
        base_config_path = Path(__file__).resolve().parent / "base_configs" / "base_fold.json"
        self.base_config = self.load_config(base_config_path)

        vertices_path = (
            Path(__file__).resolve().parent / "assets" / "meshes" / "a4_subdivision_outline.obj"
        )
        self.vertices = load_obj_vertices(vertices_path)

    def set_parameters(self):
        """No scenario parameters here, the ranges are hardcoded in the motion helpers."""

    def generate(self, save_path):
        """Randomise the folded point and build the matching handle and roller motions."""
        config = self.new_config()

        # Load vertices
        vertices = self.vertices
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) / 2

        # Pick a random vertex
        handle_idx = random.randint(0, len(vertices) - 1)
        v = vertices[handle_idx]

        # Assign handle
        config["handles"][0]["nodes"] = [handle_idx]

        # Generate a motion toward opposite side
        fold_motion, dir_xy = generate_fold_motion(v, center)

        # Insert the motion before roller motion
        config["motions"].insert(0, fold_motion)

        roller_motion = generate_roller_motion(
            dir_xy=dir_xy,
            center=center,
            start_offset=0.35,
            end_offset=0.35,
            z=0.057,
            t_start=3,
            t_end=6,
        )

        # The roller obstacle of the base config refers to motion 1
        config["motions"].insert(1, roller_motion)

        # Set handle to use this motion
        config["handles"][0]["motion"] = 0

        self.save_config(config, save_path)
