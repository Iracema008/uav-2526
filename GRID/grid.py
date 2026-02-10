"""Traversability grid for device-to-device communication."""

import json
from typing import List, Optional

from GRID.constants import GRID_SIZE, OBSTACLE, TRAVERSABLE


class TraversabilityGrid:
    """45×45 ft grid for device-to-device traversability mapping.
    Each cell: 0 = clear, 1 = obstacle (not traversable).
    """

    def __init__(self, width: int = GRID_SIZE, height: int = GRID_SIZE):
        self.width = width
        self.height = height
        self._cells: List[int] = [TRAVERSABLE] * (width * height)

    def get(self, x: int, y: int) -> int:
        """Get cell value: 0=clear, 1=obstacle."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Cell ({x},{y}) out of bounds")
        return self._cells[y * self.width + x]

    def set(self, x: int, y: int, value: int):
        """Set cell: 0=clear, 1=obstacle."""
        if value not in (TRAVERSABLE, OBSTACLE):
            raise ValueError(f"Cell value must be 0 or 1, got {value}")
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Cell ({x},{y}) out of bounds")
        self._cells[y * self.width + x] = value

    def mark_obstacle(self, x: int, y: int):
        """Mark cell as not traversable."""
        self.set(x, y, OBSTACLE)

    def mark_clear(self, x: int, y: int):
        """Mark cell as clear."""
        self.set(x, y, TRAVERSABLE)

    def to_dict(self) -> dict:
        """Package for JSON serialization (full 2D array)."""
        grid_2d = [
            [self._cells[y * self.width + x] for x in range(self.width)]
            for y in range(self.height)
        ]
        return {
            "version": 1,
            "width": self.width,
            "height": self.height,
            "grid": grid_2d,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to JSON string for transmission."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, file_path: str):
        """Save grid to a separate JSON file. Each row on one line, cells compact."""
        data = self.to_dict()
        # Format grid: each row on one line, 0s together (no newlines within rows)
        row_lines = [
            json.dumps(row, separators=(",", ":"))
            for row in data["grid"]
        ]
        grid_str = "[\n  " + ",\n  ".join(row_lines) + "\n]"
        output = (
            "{\n"
            f'  "version": {data["version"]},\n'
            f'  "width": {data["width"]},\n'
            f'  "height": {data["height"]},\n'
            f'  "grid": {grid_str}\n'
            "}\n"
        )
        with open(file_path, "w") as f:
            f.write(output)

    @classmethod
    def from_dict(cls, data: dict) -> "TraversabilityGrid":
        """Depackage from dict (e.g. after json.load)."""
        w, h = data["width"], data["height"]
        grid_2d = data["grid"]
        expected = w * h
        flat = [cell for row in grid_2d for cell in row]
        if len(flat) != expected:
            raise ValueError(
                f"Expected {expected} cells, got {len(flat)}"
            )
        g = cls(width=w, height=h)
        g._cells = [0 if c == 0 else 1 for c in flat]
        return g

    @classmethod
    def from_json(cls, json_str: str) -> "TraversabilityGrid":
        """Depackage from JSON string received from other device."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, file_path: str) -> "TraversabilityGrid":
        """Load grid from a separate JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
