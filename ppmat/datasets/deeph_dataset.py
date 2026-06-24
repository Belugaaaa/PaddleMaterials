# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import time
from typing import Dict
from typing import Tuple

import numpy as np
import paddle
import torch

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.utils import logger

try:
    from deeph import DeepHKernel
    from deeph import get_config
    from deeph import get_graph
except ImportError:
    DeepHKernel = None
    get_config = None
    get_graph = None


def _require_deeph_dependency():
    if DeepHKernel is None or get_config is None or get_graph is None:
        raise ImportError(
            "DeepHDataset requires the upstream DeepH package for graph "
            "construction. Install the DeepH data-processing dependency and "
            "make sure `import deeph` works before using DeepHDataset."
        )


def _as_tuple(config_files) -> Tuple[str, ...]:
    if isinstance(config_files, str):
        return (config_files,)
    return tuple(config_files)


def _torch_to_paddle(tensor: torch.Tensor, dtype: str) -> paddle.Tensor:
    return paddle.to_tensor(tensor.detach().cpu().numpy(), dtype=dtype)


def _list_structure_folders(raw_data_dir: str, interface: str, nums: int | None = None):
    marker = "rc.h5" if interface == "h5" else "rc.npz"
    folder_list = []
    for root, _, files in os.walk(raw_data_dir):
        if marker in files:
            folder_list.append(root)
    folder_list = sorted(folder_list)
    if nums is not None:
        folder_list = folder_list[:nums]
    return folder_list


class DeepHData(Data):
    """Geometric data object with DeepH LCMP batching semantics."""

    def __cat_dim__(self, key, value):
        if key in {
            "sub_atom_idx",
            "sub_edge_idx",
            "sub_edge_ang",
            "sub_index",
        }:
            return 0
        return super().__cat_dim__(key, value)

    def __inc__(self, key, value):
        if key == "sub_atom_idx":
            return self.num_nodes
        if key == "sub_edge_idx":
            return int(self.edge_attr.shape[0])
        if key == "sub_index":
            return int(self.edge_attr.shape[0]) * 2
        return super().__inc__(key, value)

    @staticmethod
    def collate_fn(batch):
        return Batch.from_data_list(
            batch,
            exclude_keys=[
                "structure_lattice",
                "structure_frac_coords",
                "structure_atomic_numbers",
                "structure_folder",
            ],
        )


def _load_structure_arrays(folder: str):
    lattice = np.loadtxt(os.path.join(folder, "lat.dat")).T
    atom_types = np.loadtxt(os.path.join(folder, "element.dat")).astype(int).tolist()
    cart_coords = np.loadtxt(os.path.join(folder, "site_positions.dat")).T
    frac_coords = cart_coords @ np.linalg.inv(lattice)
    return {
        "lattice": lattice,
        "atom_types": atom_types,
        "cart_coords": cart_coords,
        "frac_coords": frac_coords,
    }


class DeepHDataset(paddle.io.Dataset):
    """DeepH dataset adapter for PaddleMaterials.

    This adapter intentionally reuses DeepH's mature graph-building path so the
    PaddleMaterials integration can focus on the standard trainer / predictor
    / sampler stack first. The dataset exposes PaddleMaterials geometric
    `Data` objects and keeps the LCMP subgraph metadata required by DeepH.
    """

    _CACHE: Dict[Tuple[Tuple[str, ...], int | None, int], Dict] = {}
    _STRUCTURE_CACHE: Dict[str, Dict] = {}

    def __init__(
        self,
        config_files,
        split: str,
        nums: int | None = None,
        split_seed: int | None = None,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test.")

        self.config_files = _as_tuple(config_files)
        self.split = split
        self.nums = nums
        self.split_seed = int(split_seed) if split_seed is not None else None

        _require_deeph_dependency()
        shared = self._load_shared()
        self.dataset = shared["dataset"]
        self.folder_list = shared["folder_list"]
        self.indices = shared["split_indices"][split]
        self.info = shared["info"]

    def _load_shared(self) -> Dict:
        cache_key = (
            self.config_files,
            self.nums,
            self.split_seed or -1,
        )
        shared = self._CACHE.get(cache_key)
        if shared is not None:
            return shared

        config = get_config(list(self.config_files))
        config.set("basic", "tb_writer", "False")
        kernel = DeepHKernel(config)
        folder_list = _list_structure_folders(
            config.get("basic", "raw_dir"),
            config.get("basic", "interface"),
            self.nums,
        )

        dataset, dataset_info = self._load_or_build_graphs(config, kernel, folder_list)

        kernel.spinful = dataset_info["spinful"]
        kernel.index_to_Z = dataset_info["index_to_Z"]
        kernel.Z_to_index = dataset_info["Z_to_index"]
        kernel.num_species = len(dataset_info["index_to_Z"])
        if kernel.target not in {"E_ij", "E_i"}:
            dataset = kernel.make_mask(dataset)

        dataset_size = len(dataset)
        train_size = int(config.getfloat("train", "train_ratio") * dataset_size)
        val_size = int(config.getfloat("train", "val_ratio") * dataset_size)
        test_size = int(config.getfloat("train", "test_ratio") * dataset_size)

        seed = (
            self.split_seed
            if self.split_seed is not None
            else config.getint("basic", "seed", fallback=42)
        )
        rng = np.random.RandomState(seed)
        indices = list(range(dataset_size))
        rng.shuffle(indices)
        split_indices = {
            "train": indices[:train_size],
            "val": indices[train_size : train_size + val_size],
            "test": indices[
                train_size + val_size : train_size + val_size + test_size
            ],
        }

        shared = {
            "dataset": dataset,
            "folder_list": folder_list,
            "split_indices": split_indices,
            "info": {
                "num_species": kernel.num_species,
                "spinful": kernel.spinful,
                "dataset_size": dataset_size,
                "config_files": list(self.config_files),
            },
        }
        self._CACHE[cache_key] = shared
        return shared

    def _graph_cache_path(self, config) -> str:
        dataset_name = config.get("basic", "dataset_name")
        interface = config.get("basic", "interface")
        num_l = config.getint("network", "num_l")
        radius = config.getfloat("graph", "radius")
        max_num_nbr = config.getint("graph", "max_num_nbr")
        suffix = f"{radius}r{max_num_nbr}mn"
        if config.getboolean("graph", "create_from_DFT", fallback=True):
            suffix = "FromDFT"
        nums_suffix = "all" if self.nums is None else f"n{self.nums}"
        return os.path.join(
            config.get("basic", "graph_dir"),
            f"PPMatDeepHGraph-{interface}-{dataset_name}-{num_l}l-{suffix}-{nums_suffix}.pt",
        )

    def _load_or_build_graphs(self, config, kernel, folder_list):
        cache_path = self._graph_cache_path(config)
        os.makedirs(config.get("basic", "graph_dir"), exist_ok=True)

        if os.path.exists(cache_path):
            loaded = torch.load(cache_path, weights_only=False)
            return loaded["graphs"], loaded["info"]

        begin = time.time()
        graphs = [self._build_graph_from_folder(folder, config, kernel) for folder in folder_list]
        index_to_Z, Z_to_index = self._element_statistics(graphs)
        spinful = bool(graphs[0].spinful)
        for graph in graphs:
            assert spinful == graph.spinful

        info = {
            "spinful": spinful,
            "index_to_Z": index_to_Z,
            "Z_to_index": Z_to_index,
        }
        torch.save({"graphs": graphs, "info": info}, cache_path)
        logger.info(
            f"Finish building PaddleMaterials graph cache with {len(graphs)} structures, "
            f"cost {time.time() - begin:.0f} seconds"
        )
        return graphs, info

    def _build_graph_from_folder(self, folder: str, config, kernel):
        structure_info = self._load_structure(folder)
        structure = structure_info["structure"]
        cart_coords = torch.tensor(
            np.asarray(structure.cart_coords),
            dtype=torch.get_default_dtype(),
        )
        frac_coords = torch.tensor(
            np.asarray(structure.frac_coords),
            dtype=torch.get_default_dtype(),
        )
        numbers = torch.tensor(structure.atomic_numbers)
        lattice = torch.tensor(
            np.asarray(structure.lattice.matrix),
            dtype=torch.get_default_dtype(),
        )
        huge_structure = kernel.target == "E_ij"
        return get_graph(
            cart_coords,
            frac_coords,
            numbers,
            os.path.basename(folder),
            r=config.getfloat("graph", "radius"),
            max_num_nbr=config.getint("graph", "max_num_nbr"),
            numerical_tol=1e-8,
            lattice=lattice,
            default_dtype_torch=torch.get_default_dtype(),
            tb_folder=folder,
            interface=config.get("basic", "interface"),
            num_l=config.getint("network", "num_l"),
            create_from_DFT=config.getboolean("graph", "create_from_DFT", fallback=True),
            if_lcmp_graph=kernel.if_lcmp_graph,
            separate_onsite=kernel.separate_onsite,
            target=kernel.target,
            huge_structure=huge_structure,
            if_new_sp=kernel.new_sp,
        )

    @staticmethod
    def _element_statistics(data_list):
        index_to_Z, _ = torch.unique(data_list[0].x, sorted=True, return_inverse=True)
        Z_to_index = torch.full((100,), -1, dtype=torch.int64)
        Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))
        for data in data_list:
            data.x = Z_to_index[data.x]
        return index_to_Z, Z_to_index

    def _load_structure(self, folder: str) -> Dict:
        cached = self._STRUCTURE_CACHE.get(folder)
        if cached is not None:
            return cached

        arrays = _load_structure_arrays(folder)
        structure = BuildStructure(
            format="array",
            primitive=False,
            niggli=False,
            canocial=True,
        )(
            {
                "lattice": arrays["lattice"],
                "frac_coords": arrays["frac_coords"],
                "atom_types": arrays["atom_types"],
            }
        )
        cached = {
            "folder": folder,
            "structure": structure,
            "lattice": np.asarray(structure.lattice.matrix, dtype=np.float32),
            "frac_coords": np.asarray(structure.frac_coords, dtype=np.float32),
            "cart_coords": np.asarray(structure.cart_coords, dtype=np.float32),
            "atomic_numbers": np.asarray(structure.atomic_numbers, dtype=np.int64),
        }
        self._STRUCTURE_CACHE[folder] = cached
        return cached

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Data:
        graph_index = self.indices[index]
        graph = self.dataset[graph_index]
        folder = self.folder_list[graph_index]
        structure_info = self._load_structure(folder)

        data = DeepHData(
            x=_torch_to_paddle(graph.x, "int64"),
            edge_index=_torch_to_paddle(graph.edge_index, "int64"),
            edge_attr=_torch_to_paddle(graph.edge_attr, "float32"),
            label=_torch_to_paddle(graph.label, "float32"),
            mask=_torch_to_paddle(graph.mask.bool(), "bool"),
            pos=paddle.to_tensor(structure_info["cart_coords"], dtype="float32"),
            num_nodes=int(graph.x.shape[0]),
        )

        if hasattr(graph, "subgraph_dict") and graph.subgraph_dict is not None:
            subgraph_dict = graph.subgraph_dict
        elif hasattr(graph, "subgraph") and graph.subgraph is not None:
            subgraph = graph.subgraph
            subgraph_dict = {
                "subgraph_atom_idx": subgraph[0],
                "subgraph_edge_idx": subgraph[1],
                "subgraph_edge_ang": subgraph[2],
                "subgraph_index": subgraph[3],
            }
        else:
            raise ValueError("DeepH sample does not contain LCMP subgraph metadata.")

        data.sub_atom_idx = _torch_to_paddle(
            subgraph_dict["subgraph_atom_idx"], "int64"
        )
        data.sub_edge_idx = _torch_to_paddle(
            subgraph_dict["subgraph_edge_idx"], "int64"
        )
        data.sub_edge_ang = _torch_to_paddle(
            subgraph_dict["subgraph_edge_ang"], "float32"
        )
        data.sub_index = _torch_to_paddle(subgraph_dict["subgraph_index"], "int64")
        data.structure_lattice = paddle.to_tensor(
            structure_info["lattice"], dtype="float32"
        )
        data.structure_frac_coords = paddle.to_tensor(
            structure_info["frac_coords"], dtype="float32"
        )
        data.structure_atomic_numbers = paddle.to_tensor(
            structure_info["atomic_numbers"], dtype="int64"
        )
        data.structure_folder = structure_info["folder"]
        return data
