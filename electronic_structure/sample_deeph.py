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

import argparse
import os.path as osp
import sys

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf

from ppmat.sampler import DeepHSampler


def main():
    parser = argparse.ArgumentParser(description="Export DeepH predictions through the sampler interface.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sampler = DeepHSampler(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    result = sampler.sample_by_dataloader(cfg["Dataset"][args.split], save_path=args.save_path)
    print({"split": args.split, "prediction_shape": list(result.shape)})


if __name__ == "__main__":
    main()
