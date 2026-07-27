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

"""Readers for DeepH processed Hamiltonian and local-coordinate files."""

import json
import os

import numpy as np


def load_orbital_counts(path):
    """Return the number of atomic orbitals described on each input line."""
    with open(path) as file:
        orbital_types = [list(map(int, line.split())) for line in file if line.strip()]
    return [
        sum(2 * orbital + 1 for orbital in atom_orbitals)
        for atom_orbitals in orbital_types
    ]


def _decode_matrix_keys(npz_path, dtype):
    matrices = {}
    with np.load(npz_path) as matrix_file:
        for key_string, value in matrix_file.items():
            translation_x, translation_y, translation_z, atom_i, atom_j = json.loads(
                key_string
            )
            key = (
                translation_x,
                translation_y,
                translation_z,
                atom_i - 1,
                atom_j - 1,
            )
            matrices[key] = np.asarray(value, dtype=dtype)
    return matrices


def load_deeph_matrices(folder, dtype, include_hamiltonian=True):
    """Load orbital counts, local rotations, and optional Hamiltonian blocks."""
    orbital_counts = load_orbital_counts(os.path.join(folder, "orbital_types.dat"))
    rotations = _decode_matrix_keys(os.path.join(folder, "rc.npz"), dtype)
    hamiltonians = None
    if include_hamiltonian:
        hamiltonians = _decode_matrix_keys(os.path.join(folder, "rh.npz"), dtype)
    return orbital_counts, rotations, hamiltonians
