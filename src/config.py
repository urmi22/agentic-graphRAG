import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
