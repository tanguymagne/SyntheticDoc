import random
from pathlib import Path

from .base_config_generator import ConfigGenerator


class FallOnBallConfigGenerator(ConfigGenerator):
    """Drop a sheet on a few spheres lying on the ground."""

    def set_configs(self):
        """Load the fall base config and the sphere obstacle config."""
        base_config_path = Path(__file__).resolve().parent / "base_configs" / "base_fall.json"
        self.base_config = self.load_config(base_config_path)

        sphere_config_path = Path(__file__).resolve().parent / "base_configs" / "object_ball.json"
        self.sphere_config = self.load_config(sphere_config_path)

    def set_parameters(self):
        """Set the friction, obstacle count, scale, position and gravity ranges."""
        self.MIN_FRICTION = 0.2
        self.MAX_FRICTION = 0.5

        self.MIN_N_OBSTACLES = 1
        self.MAX_N_OBSTACLES = 3

        self.MIN_SCALE = 0.01
        self.MAX_SCALE = 0.05

        self.MIN_X_TRANSLATE = -0.05
        self.MAX_X_TRANSLATE = 0.05

        self.MIN_Y_TRANSLATE = -0.15
        self.MAX_Y_TRANSLATE = 0.15

        self.MIN_GRAVITY = -9.8
        self.MAX_GRAVITY = -1000

    def generate(self, save_path):
        """Randomise the friction, the gravity and the position, scale and number of the spheres."""
        config = self.new_config()

        obs_friction = random.uniform(self.MIN_FRICTION, self.MAX_FRICTION)
        config["obs_friction"] = obs_friction
        config["gravity"] = [0, 0, random.uniform(self.MIN_GRAVITY, self.MAX_GRAVITY)]

        obs_number = random.randint(self.MIN_N_OBSTACLES, self.MAX_N_OBSTACLES)
        # Set random obstacles
        for _ in range(obs_number):
            obs_config = self.copy_config(self.sphere_config)

            # Set a random scale for the obstacle
            scale = random.uniform(self.MIN_SCALE, self.MAX_SCALE)
            obs_config["transform"]["scale"] = scale

            # Set a random position for the obstacle
            obs_config["transform"]["translate"][0] = random.uniform(
                self.MIN_X_TRANSLATE, self.MAX_X_TRANSLATE
            )
            obs_config["transform"]["translate"][1] = random.uniform(
                self.MIN_Y_TRANSLATE, self.MAX_Y_TRANSLATE
            )

            # Sink the sphere into the ground by a random amount to vary how much of it sticks out
            obs_config["transform"]["translate"][2] = random.uniform(-scale, scale)
            config["obstacles"].append(obs_config)

        self.save_config(config, save_path)
