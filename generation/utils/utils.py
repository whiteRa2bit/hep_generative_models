from pathlib import Path
from os.path import dirname, abspath


def get_project_root() -> Path:
    """Returns project root folder."""
    return dirname(dirname(abspath(__file__)))
