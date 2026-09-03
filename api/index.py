"""Vercel entry point for the existing FastAPI application."""

import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

spec = importlib.util.spec_from_file_location(
    "project_api",
    ROOT_DIR / "api.py",
)

if spec is None or spec.loader is None:
    raise RuntimeError("Impossible de charger api.py")

project_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_api)

app = project_api.app
