# GRID Module – Overview and Reference

A small Python package for sharing traversability maps between devices over JSON. It models a 45×45 ft area, with each cell marked as either clear or obstacle.

---

## 2. Package Structure

```
GRID/
├── __init__.py      # Package exports
├── constants.py     # GRID_SIZE, TRAVERSABLE, OBSTACLE
├── grid.py          # TraversabilityGrid class
├── grid_data.json   # Sample / persisted grid data
└── DRAFT-HOW-TO-USE.md        # This file
```

---

## 3. Components

### 3.1 Constants (`constants.py`)

| Constant       | Value | Meaning           |
|----------------|-------|-------------------|
| `GRID_SIZE`    | 45    | 45×45 ft grid     |
| `TRAVERSABLE`  | 0     | Clear cell        |
| `OBSTACLE`     | 1     | Not traversable   |

### 3.2 TraversabilityGrid Class (`grid.py`)

| Method               | Purpose                                             |
|----------------------|-----------------------------------------------------|
| `__init__(width, height)` | Create grid, default 45×45                  |
| `get(x, y)`          | Get cell value: 0 or 1                              |
| `set(x, y, value)`   | Set cell to 0 or 1                                 |
| `mark_obstacle(x, y)`| Set cell to 1                                      |
| `mark_clear(x, y)`   | Set cell to 0                                      |
| `to_dict()`          | Convert to dict for JSON                            |
| `to_json(indent)`    | Serialize to string for transmission               |
| `save(file_path)`    | Save to JSON file (one line per row)               |
| `from_dict(data)`    | Build grid from dict                                |
| `from_json(str)`     | Build grid from JSON string                         |
| `load(file_path)`    | Load grid from JSON file                            |

### 3.3 JSON Schema

```json
{
  "version": 1,
  "width": 45,
  "height": 45,
  "grid": [
    [0,0,0,0,...],
    [0,0,1,0,...],
    ...
  ]
}
```

- **version**: Protocol version (for future changes)
- **width**, **height**: Grid dimensions
- **grid**: 2D array; `grid[y][x]` = cell at (x, y). One line per row when saved.

### 3.4 Coordinate Convention

- `(x, y)` = horizontal feet, vertical feet
- Origin at `(0, 0)`
- Valid range: `0 ≤ x < width`, `0 ≤ y < height`
- Internal storage: row-major, index `i = y * width + x`

---

## 4. Usage Patterns

### Device A (mapper): build, save, send

```python
from GRID import TraversabilityGrid, OBSTACLE

grid = TraversabilityGrid()
grid.mark_obstacle(10, 5)
grid.mark_obstacle(22, 30)

# Save to file
grid.save("GRID/grid_data.json")

# Or serialize for transport
json_str = grid.to_json()
# send(json_str)  # e.g., socket, MQTT, HTTP
```

### Device B (receiver): load, interpret

```python
from GRID import TraversabilityGrid, OBSTACLE, TRAVERSABLE

# From file
grid = TraversabilityGrid.load("GRID/grid_data.json")

# Or from received JSON string
# grid = TraversabilityGrid.from_json(received_string)

if grid.get(10, 5) == OBSTACLE:
    print("(10,5) is not traversable")
if grid.get(0, 0) == TRAVERSABLE:
    print("(0,0) is clear")
```

---

## 5. Design Choices (for future changes)

1. **Flat internal storage** – `_cells` is a 1D list; JSON uses 2D for clarity.
2. **Version field** – Allows adding formats or fields later without breaking readers.
3. **Full 2D grid** – Straightforward, no compression; suitable as a baseline.
4. **Per-row file formatting** – One row per line for easier diffs and inspection.
5. **Validation in `from_dict`** – Checks total cell count before accepting data.

---

## 6. Future Extensions

| Area               | Idea                                                       |
|--------------------|------------------------------------------------------------|
| **Sparse format**  | Store only obstacles as `[[x1,y1],[x2,y2],...]` for smaller payloads |
| **Timestamp**      | Add `"timestamp"` when the grid was created                 |
| **Origin/units**   | Add `"origin"` (e.g., GPS) and `"units"` (feet/meters)      |
| **Path planning**  | Use grid as input to A* or similar pathfinding              |
| **Vision integration** | Map `testing-camera/grid_test_fixed.py` outputs to grid cells |
| **Compression**    | Use run-length encoding for long sequences of 0s or 1s      |
| **Merge**          | Add `merge(other_grid)` to combine maps from different devices |
| **Diff**           | Add `diff(other_grid)` for incremental updates              |

---

## 7. Integration Points in the Repo

- **Vision / mapping**: `vision/` (OAK-D, ArUco) → obstacles into `TraversabilityGrid`
- **JSON utilities**: `vision/common/utils/json_utils.py` → read config, then pass dict to `from_dict`
- **Camera grid**: `testing-camera/grid_test_fixed.py` uses 56px = 1 ft; decide how to map pixels to grid (x, y)

---

## 8. Quick Reference

```
Import:     from GRID import TraversabilityGrid, GRID_SIZE, OBSTACLE, TRAVERSABLE
Create:     grid = TraversabilityGrid()
Modify:     grid.mark_obstacle(x, y)  /  grid.mark_clear(x, y)
Query:      grid.get(x, y)  →  0 or 1
Serialize:  grid.to_json()  /  grid.save(path)
Deserialize: TraversabilityGrid.from_json(s)  /  TraversabilityGrid.load(path)
```
