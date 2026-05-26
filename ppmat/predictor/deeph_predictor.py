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

import json
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils import save_load


class DeepHPredictor:
    """Predict Hamiltonian tensors from a configured DeepH dataset split."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        paddle.set_device(device)

        config = OmegaConf.load(config_path)
        self.config = OmegaConf.to_container(config, resolve=True)
        self.model = build_model(self.config["Model"])
        save_load.load_pretrain(self.model, checkpoint_path)
        self.model.eval()

    def predict_split(
        self,
        split: str = "test",
        save_path: Optional[str] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected train/val/test.")

        dataloader = build_dataloader(self.config["Dataset"][split])
        pred_chunks = []
        label_chunks = []
        mask_chunks = []

        with paddle.no_grad():
            for batch in dataloader:
                pred_dict = self.model.predict(batch)
                pred = pred_dict["label"]
                pred_chunks.append(pred.numpy())
                if "label" in batch:
                    label_chunks.append(batch["label"].numpy())
                if "mask" in batch:
                    mask_chunks.append(batch["mask"].numpy())

        result = {
            "split": split,
            "num_batches": len(dataloader),
            "prediction": np.concatenate(pred_chunks, axis=0),
        }
        if label_chunks:
            result["label"] = np.concatenate(label_chunks, axis=0)
        if mask_chunks:
            result["mask"] = np.concatenate(mask_chunks, axis=0)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, **result)
            logger.info(f"DeepH predictions saved to {save_path}")

        return result

    def summarize_split(
        self,
        split: str = "test",
        save_path: Optional[str] = None,
    ):
        result = self.predict_split(split=split, save_path=None)
        summary = {
            "split": split,
            "num_samples": int(result["prediction"].shape[0]),
            "prediction_shape": list(result["prediction"].shape),
        }
        if "label" in result and "mask" in result:
            mask = result["mask"].astype(bool)
            diff = np.square(result["prediction"] - result["label"])
            summary["masked_mse"] = float(diff[mask].mean()) if mask.any() else None

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(summary, indent=2))
            logger.info(f"DeepH prediction summary saved to {save_path}")
        return summary
