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

import json
import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import logger
from ppmat.utils.crystal import lattices_to_params_shape_numpy
from ppmat.utils.misc import is_equal


class DeepHDataset(Dataset):
    """DeepH Hamiltonian dataset.

    Each sample is a DeepH processed folder containing crystal geometry,
    local-coordinate matrices, and Hamiltonian labels. The dataset follows the
    PaddleMaterials dataset pattern: read raw sample folders, build structures
    and graphs, cache converted graph objects, and return collatable samples.
    """

    _STRUCTURE_CACHE: Dict[str, Dict] = {}

    def __init__(
        self,
        split: str,
        config: Dict,
        nums: int | None = None,
        split_seed: int | None = None,
        build_structure_cfg: Dict[str, Any] | None = None,
        build_graph_cfg: Dict[str, Any] | None = None,
        cache_path: str | None = None,
        overwrite: bool = False,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test.")

        self.split = split
        self.config = (
            OmegaConf.to_container(config, resolve=True)
            if OmegaConf.is_config(config)
            else config
        )
        self.basic_cfg = self.config["basic"]
        self.graph_cfg = self.config["graph"]
        self.train_cfg = self.config["train"]
        self.hyperparameter_cfg = self.config["hyperparameter"]
        self.network_cfg = self.config["network"]
        self.nums = nums
        self.split_seed = split_seed
        self.build_structure_cfg = build_structure_cfg or {
            "format": "array",
            "primitive": False,
            "niggli": False,
            "canocial": True,
        }
        self.build_graph_cfg = build_graph_cfg
        self.overwrite = overwrite
        self.structure_graph_converter = None
        if not self.graph_cfg.get("create_from_dft", True):
            self.structure_graph_converter = build_graph_converter(
                self.get_graph_converter_cfg()
            )
        self.deeph_graph_converter = build_graph_converter(
            self.get_deeph_graph_converter_cfg()
        )

        self.folder_list, self.num_samples = self.read_data(
            self.basic_cfg["raw_dir"], self.basic_cfg["interface"], nums
        )
        if self.num_samples == 0:
            raise ValueError(
                "No DeepH samples were found under "
                f"{self.basic_cfg['raw_dir']!r}. Expected folders containing rc.npz."
            )
        logger.info(
            f"Load {self.num_samples} DeepH samples from {self.basic_cfg['raw_dir']}"
        )

        self.cache_path = cache_path or self.get_cache_path()
        logger.info(f"Cache path: {self.cache_path}")

        graph_cache_path = osp.join(self.cache_path, "graphs.pkl")
        cfg_cache_path = osp.join(self.cache_path, "dataset_cfg.pkl")
        cache_cfg = {
            "folders": self.folder_list,
            "nums": self.nums,
            "build_structure_cfg": self.build_structure_cfg,
            "build_graph_cfg": self.get_graph_converter_cfg(),
            "interface": self.basic_cfg["interface"],
            "target": self.basic_cfg["target"],
            "num_l": self.network_cfg["num_l"],
            "dtype": self.hyperparameter_cfg["dtype"],
            "radius": self.graph_cfg["radius"],
            "max_num_nbr": self.graph_cfg["max_num_nbr"],
        }

        rebuild_cache = True
        if osp.exists(graph_cache_path) and not overwrite:
            try:
                cached_cfg = self.load_from_cache(cfg_cache_path)
                if is_equal(cached_cfg, cache_cfg):
                    cached_data = self.load_from_cache(graph_cache_path)
                    self.graphs = cached_data["graphs"]
                    self.dataset_info = cached_data["info"]
                    rebuild_cache = False
                else:
                    logger.warning("DeepH cache config differs. Rebuilding graphs.")
                    self.graphs, self.dataset_info = self.build_graphs()
            except Exception as error:
                logger.warning(error)
                logger.warning("Failed to load DeepH graph cache. Rebuilding graphs.")
                self.graphs, self.dataset_info = self.build_graphs()
        else:
            self.graphs, self.dataset_info = self.build_graphs()

        if rebuild_cache:
            self.save_to_cache(cfg_cache_path, cache_cfg)
            self.save_to_cache(
                graph_cache_path, {"graphs": self.graphs, "info": self.dataset_info}
            )

        if self.basic_cfg["target"] != "hamiltonian":
            raise NotImplementedError(
                "DeepHDataset currently supports the validated hamiltonian target."
            )
        self.graphs = self.make_hamiltonian_label(
            self.graphs,
            orbital=self.basic_cfg["orbital"],
            spinful=self.dataset_info["spinful"],
            index_to_Z=self.dataset_info["index_to_Z"],
        )

        dataset_size = len(self.graphs)
        train_size = int(self.train_cfg["train_ratio"] * dataset_size)
        val_size = int(self.train_cfg["val_ratio"] * dataset_size)
        test_size = int(self.train_cfg["test_ratio"] * dataset_size)
        seed = (
            self.split_seed
            if self.split_seed is not None
            else self.basic_cfg.get("seed", 42)
        )
        indices = list(range(dataset_size))
        np.random.RandomState(seed).shuffle(indices)
        split_indices = {
            "train": indices[:train_size],
            "val": indices[train_size : train_size + val_size],
            "test": indices[train_size + val_size : train_size + val_size + test_size],
        }
        self.indices = split_indices[split]
        self.info = {
            "num_species": len(self.dataset_info["index_to_Z"]),
            "spinful": self.dataset_info["spinful"],
            "dataset_size": dataset_size,
        }

    def read_data(self, raw_data_dir: str, interface: str, nums: int | None = None):
        """Read DeepH processed sample folders."""
        if interface != "npz":
            raise NotImplementedError(
                "DeepHDataset currently supports the validated npz Hamiltonian format."
            )
        folder_list = sorted(
            root for root, _, files in os.walk(raw_data_dir) if "rc.npz" in files
        )
        if nums is not None:
            folder_list = folder_list[:nums]
        return folder_list, len(folder_list)

    def get_cache_path(self):
        radius = self.graph_cfg["radius"]
        max_num_nbr = self.graph_cfg["max_num_nbr"]
        suffix = f"{radius}r{max_num_nbr}mn"
        if self.graph_cfg.get("create_from_dft", True):
            suffix = "FromDFT"
        nums_suffix = "all" if self.nums is None else f"n{self.nums}"
        return osp.join(
            self.basic_cfg["graph_dir"],
            "PPMatDeepHGraph-"
            f"{self.basic_cfg['interface']}-{self.basic_cfg['dataset_name']}-"
            f"{self.network_cfg['num_l']}l-{suffix}-{nums_suffix}",
        )

    def build_graphs(self):
        """Build PaddleMaterials graph objects from DeepH sample folders."""
        graphs = [self.build_graph(folder) for folder in self.folder_list]
        index_to_Z = np.unique(graphs[0].x).astype(np.int64)
        Z_to_index = np.full((100,), -1, dtype=np.int64)
        Z_to_index[index_to_Z] = np.arange(len(index_to_Z), dtype=np.int64)
        for graph in graphs:
            graph.x = Z_to_index[graph.x]

        spinful = bool(graphs[0].spinful)
        for graph in graphs:
            assert spinful == graph.spinful
        return graphs, {
            "spinful": spinful,
            "index_to_Z": index_to_Z,
            "Z_to_index": Z_to_index,
        }

    def build_structure(self, folder: str):
        """Build a pymatgen Structure from one DeepH processed folder."""
        cache_key = json.dumps(
            {"folder": folder, "cfg": self.build_structure_cfg}, sort_keys=True
        )
        if cache_key in self._STRUCTURE_CACHE:
            return self._STRUCTURE_CACHE[cache_key]

        lattice = np.loadtxt(osp.join(folder, "lat.dat")).T
        atom_types = np.loadtxt(osp.join(folder, "element.dat")).astype(int).tolist()
        cart_coords = np.loadtxt(osp.join(folder, "site_positions.dat")).T
        frac_coords = cart_coords @ np.linalg.inv(lattice)
        lengths, angles = lattices_to_params_shape_numpy(lattice)

        builder = BuildStructure(**self.build_structure_cfg)
        structure = BuildStructure.build_one(
            {
                "lengths": lengths,
                "angles": angles,
                "frac_coords": frac_coords,
                "atom_types": atom_types,
            },
            builder.format,
            builder.primitive,
            builder.niggli,
            builder.canocial,
        )
        data = {
            "folder": folder,
            "structure": structure,
            "lattice": np.asarray(structure.lattice.matrix, dtype=np.float32),
            "frac_coords": np.asarray(structure.frac_coords, dtype=np.float32),
            "cart_coords": np.asarray(structure.cart_coords, dtype=np.float32),
            "atomic_numbers": np.asarray(structure.atomic_numbers, dtype=np.int64),
        }
        self._STRUCTURE_CACHE[cache_key] = data
        return data

    def build_graph(self, folder: str):
        """Build one DeepH graph object."""
        structure = self.build_structure(folder)["structure"]
        converter_graph = None
        if self.structure_graph_converter is not None:
            converter_graph = self.structure_graph_converter(structure)
            if converter_graph is None:
                raise ValueError(f"Failed to build graph for DeepH structure: {folder}")
        return self.deeph_graph_converter(structure, folder, converter_graph)

    def get_graph_converter_cfg(self):
        if self.build_graph_cfg is not None:
            return self.build_graph_cfg
        return {
            "__class_name__": "FindPointsInSpheres",
            "__init_params__": {
                "cutoff": self.graph_cfg["radius"],
                "pbc": (1, 1, 1),
                "eps": 1e-8,
            },
        }

    def get_deeph_graph_converter_cfg(self):
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
        }
        return {
            "__class_name__": "DeepHGraphConverter",
            "__init_params__": {
                "radius": self.graph_cfg["radius"],
                "max_num_nbr": self.graph_cfg["max_num_nbr"],
                "default_dtype": dtype_map[self.hyperparameter_cfg["dtype"]],
                "interface": self.basic_cfg["interface"],
                "num_l": self.network_cfg["num_l"],
                "create_from_DFT": self.graph_cfg.get("create_from_dft", True),
                "if_lcmp_graph": self.graph_cfg.get("if_lcmp_graph", True),
                "separate_onsite": self.graph_cfg.get("separate_onsite", False),
                "target": self.basic_cfg["target"],
                "huge_structure": False,
                "if_new_sp": self.graph_cfg.get("new_sp", False),
            },
        }

    def make_hamiltonian_label(self, graphs, orbital, spinful, index_to_Z):
        """Attach Hamiltonian labels and masks to DeepH graph objects."""
        out_fea_len = len(orbital) * 8 if spinful else len(orbital)
        for graph in graphs:
            oij_value = graph.term_real
            if not np.all(graph.term_mask):
                raise NotImplementedError(
                    "Graph radius including hopping without calculation is not "
                    "supported."
                )

            mask = np.zeros((graph.edge_attr.shape[0], out_fea_len), dtype=np.int8)
            label = np.zeros(
                (graph.edge_attr.shape[0], out_fea_len), dtype=oij_value.dtype
            )
            atomic_number_edge_i = index_to_Z[graph.x[graph.edge_index[0]]]
            atomic_number_edge_j = index_to_Z[graph.x[graph.edge_index[1]]]

            for index_out, orbital_dict in enumerate(orbital):
                for n_m_str, orbital_pair in orbital_dict.items():
                    atomic_number_i, atomic_number_j = map(int, n_m_str.split())
                    orbital_i, orbital_j = orbital_pair
                    condition = (atomic_number_edge_i == atomic_number_i) & (
                        atomic_number_edge_j == atomic_number_j
                    )
                    if spinful:
                        value = oij_value[:, orbital_i, orbital_j]
                        mask[:, 8 * index_out : 8 * (index_out + 1)] = np.where(
                            condition[:, None], 1, 0
                        )
                        label[:, 8 * index_out : 8 * (index_out + 1)] = np.where(
                            condition[:, None], value, np.zeros_like(value)
                        )
                    else:
                        mask[:, index_out] += np.where(condition, 1, 0)
                        label[:, index_out] += np.where(
                            condition,
                            oij_value[:, orbital_i, orbital_j],
                            np.zeros(graph.edge_attr.shape[0], dtype=oij_value.dtype),
                        )

            assert len(np.where((mask != 1) & (mask != 0))[0]) == 0
            graph.mask = mask.astype(bool)
            graph.label = label
            del graph.term_mask
            del graph.term_real
        return graphs

    def save_to_cache(self, cache_path: str, data: Any):
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as file:
            pickle.dump(data, file)

    def load_from_cache(self, cache_path: str):
        if not osp.exists(cache_path):
            raise FileNotFoundError(f"No such file or directory: {cache_path}")
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    def __getitem__(self, index: int):
        graph_index = self.indices[index]
        graph = self.graphs[graph_index]
        structure_info = self.build_structure(self.folder_list[graph_index])

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

        return {
            "x": ConcatData(np.asarray(graph.x, dtype=np.int64)),
            "edge_index": ConcatData(np.asarray(graph.edge_index, dtype=np.int64)),
            "edge_attr": ConcatData(np.asarray(graph.edge_attr, dtype=np.float32)),
            "batch": ConcatData(np.zeros(graph.x.shape[0], dtype=np.int64)),
            "label": ConcatData(np.asarray(graph.label, dtype=np.float32)),
            "mask": ConcatData(np.asarray(graph.mask, dtype=bool)),
            "pos": ConcatData(
                np.asarray(structure_info["cart_coords"], dtype=np.float32)
            ),
            "sub_atom_idx": ConcatData(
                np.asarray(subgraph_dict["subgraph_atom_idx"], dtype=np.int64)
            ),
            "sub_edge_idx": ConcatData(
                np.asarray(subgraph_dict["subgraph_edge_idx"], dtype=np.int64)
            ),
            "sub_edge_ang": ConcatData(
                np.asarray(subgraph_dict["subgraph_edge_ang"], dtype=np.float32)
            ),
            "sub_index": ConcatData(
                np.asarray(subgraph_dict["subgraph_index"], dtype=np.int64)
            ),
            "structure_lattice": ConcatData(
                np.asarray(structure_info["lattice"], dtype=np.float32).reshape(1, 3, 3)
            ),
            "structure_frac_coords": ConcatData(
                np.asarray(structure_info["frac_coords"], dtype=np.float32)
            ),
            "structure_atomic_numbers": ConcatData(
                np.asarray(structure_info["atomic_numbers"], dtype=np.int64)
            ),
            "structure_folder": structure_info["folder"],
        }

    def __len__(self):
        return len(self.indices)
