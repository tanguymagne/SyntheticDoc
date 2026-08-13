import random
from pathlib import Path

from .base_config_generator import ConfigGenerator


class CurveByPullConfigGenerator(ConfigGenerator):
    """Curve a sheet by pulling one or two of its nodes upwards."""

    def set_configs(self):
        """Load the pull base config."""
        base_config_path = Path(__file__).resolve().parent / "base_configs" / "base_curve.json"
        self.base_config = self.load_config(base_config_path)

    def set_parameters(self):
        """Set the gravity, pull height and mesh resolution parameters."""
        self.MIN_GRAVITY = -9.8
        self.MAX_GRAVITY = -100

        self.MIN_PULL_HEIGHT = 0.03
        self.MAX_PULL_HEIGHT = 0.08

        self.NODE_PER_MESH = 1681

    def generate(self, save_path):
        """Randomise the pulled nodes, the gravity and the pull height."""
        config = self.new_config()

        if random.random() < 0.05:
            # Pull a single node
            config["handles"][0]["nodes"] = [random.randint(0, self.NODE_PER_MESH - 1)]
        else:
            # Pull two nodes
            config["handles"][0]["nodes"] = random.sample(range(self.NODE_PER_MESH), 2)

        # Exaggerated gravity keeps the rest of the sheet flat on the ground while it is pulled
        config["gravity"] = [0, 0, random.uniform(self.MIN_GRAVITY, self.MAX_GRAVITY)]

        random_pull_height = random.uniform(self.MIN_PULL_HEIGHT, self.MAX_PULL_HEIGHT)
        config["motions"][0][1]["transform"]["translate"][2] = random_pull_height

        self.save_config(config, save_path)
