"""Storage module for face database and access logging."""

from .face_db import FaceDatabase
from .logs import AccessLogger

__all__ = ["FaceDatabase", "AccessLogger"]
