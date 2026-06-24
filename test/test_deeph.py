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

import os
import unittest

import paddle
from omegaconf import OmegaConf

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.deeph_dataset import DeepHData
from ppmat.models.deeph import DeepHHamiltonian


class TestDeepH(unittest.TestCase):
    def test_deeph_config_can_be_loaded(self):
        config_path = os.path.join(
            "electronic_structure",
            "configs",
            "deeph",
            "deeph_graphene.yaml",
        )
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

        self.assertEqual(config["Model"]["__class_name__"], "DeepHHamiltonian")
        self.assertEqual(
            config["Dataset"]["train"]["dataset"]["__class_name__"],
            "DeepHDataset",
        )

    def test_deeph_forward(self):
        paddle.seed(42)
        model = DeepHHamiltonian(
            num_species=1,
            in_atom_fea_len=16,
            in_edge_fea_len=32,
            num_orbital=9,
            num_l=3,
            gauss_stop=6.0,
            if_exp=True,
            normalization="LayerNorm",
            target_name="label",
        )

        num_nodes = 3
        num_edges = 4
        num_sub_edges = 8
        batch = {
            "x": paddle.zeros([num_nodes], dtype="int64"),
            "edge_index": paddle.to_tensor(
                [[0, 1, 1, 2], [1, 0, 2, 1]], dtype="int64"
            ),
            "edge_attr": paddle.ones([num_edges, 1], dtype="float32"),
            "batch": paddle.zeros([num_nodes], dtype="int64"),
            "sub_atom_idx": paddle.to_tensor(
                [
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                    [0, 1],
                    [1, 0],
                    [1, 2],
                    [2, 1],
                ],
                dtype="int64",
            ),
            "sub_edge_idx": paddle.to_tensor(
                [0, 1, 2, 3, 0, 1, 2, 3], dtype="int64"
            ),
            "sub_edge_ang": paddle.zeros([num_sub_edges, 9], dtype="float32"),
            "sub_index": paddle.to_tensor(
                [0, 1, 2, 3, 4, 5, 6, 7], dtype="int64"
            ),
            "label": paddle.zeros([num_edges, 9], dtype="float32"),
            "mask": paddle.ones([num_edges, 9], dtype="bool"),
        }

        output = model(batch)

        self.assertIn("loss", output["loss_dict"])
        self.assertIn("label", output["pred_dict"])
        self.assertEqual(list(output["pred_dict"]["label"].shape), [num_edges, 9])

    def test_deeph_data_uses_default_collator(self):
        sample = DeepHData(
            x=paddle.zeros([2], dtype="int64"),
            edge_index=paddle.to_tensor([[0, 1], [1, 0]], dtype="int64"),
            edge_attr=paddle.ones([2, 1], dtype="float32"),
            label=paddle.zeros([2, 9], dtype="float32"),
            mask=paddle.ones([2, 9], dtype="bool"),
            num_nodes=2,
        )
        sample.sub_atom_idx = paddle.to_tensor([[0, 1], [1, 0]], dtype="int64")
        sample.sub_edge_idx = paddle.to_tensor([0, 1], dtype="int64")
        sample.sub_edge_ang = paddle.zeros([2, 9], dtype="float32")
        sample.sub_index = paddle.to_tensor([0, 1], dtype="int64")

        batch = DefaultCollator()([sample, sample])

        self.assertEqual(list(batch.x.shape), [4])
        self.assertEqual(batch.sub_atom_idx.numpy().tolist(), [[0, 1], [1, 0], [2, 3], [3, 2]])
        self.assertEqual(batch.sub_edge_idx.numpy().tolist(), [0, 1, 2, 3])
        self.assertEqual(batch.sub_index.numpy().tolist(), [0, 1, 4, 5])


if __name__ == "__main__":
    unittest.main()
