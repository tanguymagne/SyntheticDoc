"""Base class shared by the arcsim scenario config generators."""

import copy
import json
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


class ConfigGenerator(ABC):
    """Generate randomised arcsim configs for a single scenario.

    A generator reads the scenario base config once, then writes one randomised
    copy of it per config. Subclasses declare their randomisation ranges in
    `set_parameters` and implement `generate`.
    """

    def __init__(self):
        """Load the configs and the randomisation parameters of the scenario."""
        self.set_configs()
        self.set_parameters()

    @abstractmethod
    def set_configs(self):
        """Set the base config and object configs."""

    @abstractmethod
    def set_parameters(self):
        """Set the scenario specific randomisation parameters as attributes."""

    @abstractmethod
    def generate(self, save_path):
        """Write a single randomised config to `save_path`."""

    def new_config(self):
        """Return a fresh copy of the base config, ready to be randomised."""
        return copy.deepcopy(self.base_config)

    @staticmethod
    def load_config(config_path):
        """Load a config from `config_path` and return it as a dict."""
        with Path(config_path).open("r") as f:
            return json.load(f)

    @staticmethod
    def copy_config(config):
        """Return a deep copy of `config`."""
        return copy.deepcopy(config)

    @staticmethod
    def save_config(config, save_path):
        """Write `config` to `save_path` as JSON."""
        with Path(save_path).open("w") as f:
            json.dump(config, f, indent=4)

    def batch_generate(self, seed, num, save_dir):
        """Generate `num` configs named `00000.json`, `00001.json`, ... in `save_dir`."""
        random.seed(seed)
        np.random.seed(seed)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(num)):
            self.generate(save_dir / f"{i:05d}.json")
