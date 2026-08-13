import math
import random
from pathlib import Path

from .base_config_generator import ConfigGenerator


class FallOnRollerConfigGenerator(ConfigGenerator):
    """Drop a sheet on a few near-horizontal cylinders lying on the ground."""

    def set_configs(self):
        """Load the fall base config and the cylinder obstacle config."""
        base_config_path = Path(__file__).resolve().parent / "base_configs" / "base_fall.json"
        self.base_config = self.load_config(base_config_path)

        cylinder_config_path = (
            Path(__file__).resolve().parent / "base_configs" / "object_roller.json"
        )
        self.cylinder_config = self.load_config(cylinder_config_path)

    def set_parameters(self):
        """Set the friction, obstacle count, orientation, scale and position ranges."""
        self.MIN_FRICTION = 0.2
        self.MAX_FRICTION = 0.5

        self.MIN_N_OBSTACLES = 1
        self.MAX_N_OBSTACLES = 3

        self.MIN_DIRECTION_ANGLE = 0
        self.MAX_DIRECTION_ANGLE = 360

        self.MIN_VERTICAL_ANGLE = 85
        self.MAX_VERTICAL_ANGLE = 95

        self.MIN_SCALE = 0.01
        self.MAX_SCALE = 0.1

        self.MIN_X_TRANSLATE = -0.15
        self.MAX_X_TRANSLATE = 0.15

        self.MIN_Y_TRANSLATE = -0.2
        self.MAX_Y_TRANSLATE = 0.2

    def generate(self, save_path):
        """Randomise the friction and the pose, scale and number of the cylinders."""
        config = self.new_config()

        obs_friction = random.uniform(self.MIN_FRICTION, self.MAX_FRICTION)
        config["obs_friction"] = obs_friction
        config["end_time"] = 1
        obs_number = random.randint(self.MIN_N_OBSTACLES, self.MAX_N_OBSTACLES)

        # Set random obstacles
        for _ in range(obs_number):
            obs_config = self.copy_config(self.cylinder_config)

            # Set a random orientation for the obstacle
            rot_direction_angle = random.uniform(self.MIN_DIRECTION_ANGLE, self.MAX_DIRECTION_ANGLE)
            rot_direction_x = math.cos(math.radians(rot_direction_angle))
            rot_direction_y = math.sin(math.radians(rot_direction_angle))

            # Close to 90 degrees, so the cylinder lies almost flat on the ground
            rot_angle = random.randint(self.MIN_VERTICAL_ANGLE, self.MAX_VERTICAL_ANGLE)
            obs_config["transform"]["rotate"][0] = rot_angle
            obs_config["transform"]["rotate"][1] = rot_direction_x
            obs_config["transform"]["rotate"][2] = rot_direction_y
            obs_config["transform"]["rotate"][3] = 0

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

            # Sink the cylinder into the ground by a random amount to vary how much of it sticks out
            obs_config["transform"]["translate"][2] = random.uniform(-scale, scale)
            config["obstacles"].append(obs_config)

        self.save_config(config, save_path)
