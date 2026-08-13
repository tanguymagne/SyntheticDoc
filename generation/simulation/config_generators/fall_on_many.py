import math
import random
from pathlib import Path

from .base_config_generator import ConfigGenerator


class FallOnManyConfigGenerator(ConfigGenerator):
    """Drop a sheet on a dense jittered grid of small spheres, to crumple it rather than bend it."""

    def set_configs(self):
        """Load the fall base config and the sphere obstacle config."""
        base_config_path = Path(__file__).resolve().parent / "base_configs" / "base_fall.json"
        self.base_config = self.load_config(base_config_path)

        sphere_config_path = Path(__file__).resolve().parent / "base_configs" / "object_ball.json"
        self.sphere_config = self.load_config(sphere_config_path)

    def set_parameters(self):
        """Set the friction, sphere count, grid extent, scale, jitter and gravity ranges."""
        self.MIN_FRICTION = 0.25
        self.MAX_FRICTION = 0.5

        self.MIN_N_SPHERES = 30
        self.MAX_N_SPHERES = 150

        self.PAPER_X_MIN = -0.15
        self.PAPER_X_MAX = 0.15
        self.PAPER_Y_MIN = -0.21
        self.PAPER_Y_MAX = 0.21

        self.MIN_SCALE = 0.005
        self.MAX_SCALE = 0.01

        self.XY_JITTER_FACTOR = 0.4
        self.Z_JITTER_FACTOR = 0.5

        self.MIN_GRAVITY = -1000
        self.MAX_GRAVITY = -5000

        self.MIN_ROT_ANGLE = 0
        self.MAX_ROT_ANGLE = 360

    def generate(self, save_path):
        """Randomise the friction, the gravity and the layout of the sphere grid."""
        config = self.new_config()

        config["obs_friction"] = random.uniform(self.MIN_FRICTION, self.MAX_FRICTION)

        # Soften the paper so it keeps the creases left by the spheres
        config["cloths"][0]["materials"][0]["yield_curv"] = 50
        config["cloths"][0]["materials"][0]["weakening"] = 0.2

        config["gravity"] = [0, 0, random.uniform(self.MIN_GRAVITY, self.MAX_GRAVITY)]

        # The exaggerated gravity makes the sheet settle almost immediately
        config["end_time"] = 0.1

        n_obs = random.randint(self.MIN_N_SPHERES, self.MAX_N_SPHERES)

        # Lay the spheres out on a square grid covering the sheet
        grid_n = int(math.sqrt(n_obs))
        grid_n = max(grid_n, 1)

        xs = [
            self.PAPER_X_MIN + (self.PAPER_X_MAX - self.PAPER_X_MIN) * (i + 0.5) / grid_n
            for i in range(grid_n)
        ]
        ys = [
            self.PAPER_Y_MIN + (self.PAPER_Y_MAX - self.PAPER_Y_MIN) * (j + 0.5) / grid_n
            for j in range(grid_n)
        ]

        spacing_x = (self.PAPER_X_MAX - self.PAPER_X_MIN) / grid_n
        spacing_y = (self.PAPER_Y_MAX - self.PAPER_Y_MIN) / grid_n

        count = 0
        for x in xs:
            for y in ys:
                if count >= n_obs:
                    break

                obs = self.copy_config(self.sphere_config)

                scale = random.uniform(self.MIN_SCALE, self.MAX_SCALE)

                # Offset each sphere from its cell center, so the grid does not show in the creases
                jitter_x = random.uniform(-self.XY_JITTER_FACTOR, self.XY_JITTER_FACTOR) * spacing_x
                jitter_y = random.uniform(-self.XY_JITTER_FACTOR, self.XY_JITTER_FACTOR) * spacing_y
                jitter_z = random.uniform(-self.Z_JITTER_FACTOR, self.Z_JITTER_FACTOR) * scale

                obs["transform"]["scale"] = scale
                obs["transform"]["translate"][0] = x + jitter_x
                obs["transform"]["translate"][1] = y + jitter_y
                obs["transform"]["translate"][2] = jitter_z
                config["obstacles"].append(obs)
                count += 1

        self.save_config(config, save_path)
