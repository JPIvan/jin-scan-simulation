import numpy as np
from pathlib import Path
from typing import List, Optional

from jin_scan_simulation.polygons import GridPolygon


class GridPolygonDataset:
    def __init__(self, *, seed=0) -> None:
        self._GP_MODULE_DIR = Path(__file__).resolve().parent.parent
        self._VERTEX_DATA_DIR = self._GP_MODULE_DIR / "data" / "vertex_data"

        self._polygons = None
        self.rng = np.random.default_rng(seed=seed)

    def load_polygons(
                self,
                name: str = "run10k",
                _VERTEX_DATA_DIR: Optional[Path] = None
            ) -> None:
        """Load polygons using vertex data from datafiles.

        Args:
            name (str, optional):
                Name of dataset files.
                These should be found in the root folder under
                    ./data/vertex_data/
                Expects 1 or more datafiles with naming "<name>_<n>.npy"
                Where <name> is this parameter, and <n> is an integer.
                Defaults to "run10k".
            _VERTEX_DATA_DIR (Path, optional):
                Path to the vertex data dir if it is not in the usual location.
                Defaults to None.
                If None will use ./data/vertex_data/
        """
        data = self._load_vertex_data(
            name=name, _VERTEX_DATA_DIR=_VERTEX_DATA_DIR
        )
        self._polygons = self._build_polygons(data)
        return

    def __getitem__(self, idx: int) -> GridPolygon:
        return self._polygons[idx]

    def _load_vertex_data(
                self,
                name: str = "run10k",
                _VERTEX_DATA_DIR: Optional[Path] = None
            ) -> List[np.array]:
        """Load vertex data from existing datafiles.

        Args:
            name (str, optional):
                Name of dataset files.
                These should be found in the root folder under
                    ./data/vertex_data/
                Expects 1 or more datafiles with naming "<name>_<n>.npy"
                Where <name> is this parameter, and <n> is an integer.
                Defaults to "run10k".
            _VERTEX_DATA_DIR (Path, optional):
                Path to the vertex data dir if it is not in the usual location.
                Defaults to None.
                If None will use ./data/vertex_data/

        Returns:
            List[np.array]: A list containing one array for each datafile.
        """
        if _VERTEX_DATA_DIR is None:
            _VERTEX_DATA_DIR = self._VERTEX_DATA_DIR
        n = np.sum([name in str(path) for path in _VERTEX_DATA_DIR.iterdir()])
        data = []
        for i in range(n):
            dfile = _VERTEX_DATA_DIR.joinpath(f"{name}_{i}.npy")
            data.append(np.load(dfile))
        return data

    def _build_polygons(self, vertex_data: List[np.array]) \
            -> List[GridPolygon]:
        """Build polygons from given vertex data.

        Returns:
            List[GridPolygon]: List of constructed polygons
        """
        polygons = []
        for V in vertex_data:  # multiple polygons with same number of vertices
            for v in V:  # individual polygons
                polygons.append(
                    GridPolygon(V=v, seed=self.rng.integers(2**16))
                )  # Polygon rng will be predictible if self.rng is seeded
        return polygons
