import random
from pathlib import Path

from .base_config_generator import ConfigGenerator


class FallOnRoofConfigGenerator(ConfigGenerator):
    """Drop a sheet on a few gable (roof shaped) obstacles lying on the ground."""

    def set_configs(self):
        """Load the fall base config and the gable obstacle config."""
        base_config_path = Path(__file__).resolve().parent / "base_configs" / "base_fall.json"
        self.base_config = self.load_config(base_config_path)

        gable_config_path = Path(__file__).resolve().parent / "base_configs" / "object_roof.json"
        self.gable_config = self.load_config(gable_config_path)

    def set_parameters(self):
        """Set the friction, obstacle count, rotation, scale, position and gravity ranges."""
        self.MIN_FRICTION = 0.2
        self.MAX_FRICTION = 0.5

        self.MIN_N_OBSTACLES = 1
        self.MAX_N_OBSTACLES = 3

        self.MIN_ROTATION_ANGLE = 0
        self.MAX_ROTATION_ANGLE = 180

        self.MIN_SCALE = 0.01
        self.MAX_SCALE = 0.05

        self.MIN_X_TRANSLATE = -0.12
        self.MAX_X_TRANSLATE = 0.12

        self.MIN_Y_TRANSLATE = -0.15
        self.MAX_Y_TRANSLATE = 0.15

        self.MIN_GRAVITY = -100
        self.MAX_GRAVITY = -1000

    def generate(self, save_path):
        """Randomise the friction, the gravity and the pose, scale and number of the gables."""
        config = self.new_config()

        obs_friction = random.uniform(self.MIN_FRICTION, self.MAX_FRICTION)
        config["obs_friction"] = obs_friction
        config["gravity"] = [0, 0, random.uniform(self.MIN_GRAVITY, self.MAX_GRAVITY)]

        obs_number = random.randint(self.MIN_N_OBSTACLES, self.MAX_N_OBSTACLES)

        # Set random obstacles
        for _ in range(obs_number):
            obs_config = self.copy_config(self.gable_config)

            # Set a random scale for the obstacle
            scale = random.uniform(self.MIN_SCALE, self.MAX_SCALE)
            obs_config["transform"]["scale"] = scale

            # Spin the gable around the vertical axis, its ridge is symmetric so 180 degrees suffice
            rot_angle = random.randint(self.MIN_ROTATION_ANGLE, self.MAX_ROTATION_ANGLE)
            obs_config["transform"]["rotate"][0] = rot_angle
            obs_config["transform"]["rotate"][3] = 1

            # Set a random position for the obstacle
            obs_config["transform"]["translate"][0] = random.uniform(
                self.MIN_X_TRANSLATE, self.MAX_X_TRANSLATE
            )
            obs_config["transform"]["translate"][1] = random.uniform(
                self.MIN_Y_TRANSLATE, self.MAX_Y_TRANSLATE
            )

            # Keep the gable at or below ground level, so only its roof sticks out
            obs_config["transform"]["translate"][2] = random.uniform(-scale, 0)
            config["obstacles"].append(obs_config)

        self.save_config(config, save_path)
