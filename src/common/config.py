# config.py
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config() 