"""JoLA configuration loader."""

import os

import yaml


class JoLAConfig:
    def __init__(self, default=True, config_path=None):
        self.default = default
        self.config_path = config_path

    @classmethod
    def get_jola_config(cls, default=True, config_path=None):
        if default:
            script_dir = os.path.dirname(__file__)
            config_path = os.path.join(script_dir, "config.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            if config_path is None:
                raise ValueError("config_path must be provided when default is False")
            with open(config_path) as f:
                config = yaml.safe_load(f)
        return config
