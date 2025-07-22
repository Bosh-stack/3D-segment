import json
import pickle
from pathlib import Path
from typing import Dict, Any

import torch


class BaseDataset:
    """Minimal base dataset."""

    def __init__(self, cfg: Dict[str, Any], split: str = "train"):
        self.cfg = cfg
        self.split = split

    def __len__(self):
        raise NotImplementedError

    def get_item(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class MySetDataset(BaseDataset):
    """Lightweight adapter for custom dataset layout."""

    NAME = "myset"

    def __init__(self, cfg: Dict[str, Any], split: str = "train"):
        super().__init__(cfg, split)
        paths = cfg.get("PATHS", {})
        myset = paths.get("myset", {})
        self.paths = {
            "root": Path(myset.get("root", "/data/Open3DSG_trainset")),
            "graphs": Path(myset.get("graphs_out", "open3dsg/output/graphs/myset")) / "graphs",
            "cache": Path(myset.get("preproc_out", "open3dsg/output/preprocessed/myset")) / "cache",
            "frames": Path(myset.get("preproc_out", "open3dsg/output/preprocessed/myset")) / "frames",
        }
        graphs_file = self.paths["graphs"] / f"{split}.json"
        self.graph_db = json.loads(graphs_file.read_text()) if graphs_file.exists() else {}
        self.scan_ids = sorted(self.graph_db.keys())

    def __len__(self) -> int:
        return len(self.scan_ids)

    def _load_scene(self, idx: int) -> Dict[str, Any]:
        scan_id = self.scan_ids[idx]
        points = torch.load(self.paths["cache"] / f"{scan_id}_pc.pt")
        with open(self.paths["cache"] / f"{scan_id}_instances.pkl", "rb") as f:
            instances = pickle.load(f)
        graph = self.graph_db[scan_id]
        frames_map_path = self.paths["frames"] / f"{scan_id}.json"
        frames_map = json.loads(frames_map_path.read_text()) if frames_map_path.exists() else {}
        return {
            "scan_id": scan_id,
            "points": points,
            "instances": instances,
            "graph": graph,
            "frames": frames_map,
        }

    def get_item(self, idx: int) -> Dict[str, Any]:
        return self._load_scene(idx)
