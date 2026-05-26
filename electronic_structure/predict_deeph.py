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

from ppmat.predictor import DeepHPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict DeepH Hamiltonians from a dataset split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args()

    predictor = DeepHPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    predictor.predict_split(split=args.split, save_path=args.save_path)
    summary = predictor.summarize_split(split=args.split, save_path=args.summary_path)
    print(summary)


if __name__ == "__main__":
    main()
