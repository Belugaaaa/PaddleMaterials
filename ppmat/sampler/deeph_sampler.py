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

from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils import save_load


class DeepHSampler:
    """Unified sampler-style export for DeepH predictions.

    DeepH is not a generative model. Here `sample` means exporting model outputs
    from a configured dataset split through the common sampler namespace.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        paddle.set_device(device)
        config = OmegaConf.load(config_path)
        self.config = OmegaConf.to_container(config, resolve=True)
        self.model = build_model(self.config["Model"])
        save_load.load_pretrain(self.model, checkpoint_path)
        self.model.eval()

    def sample_by_dataloader(
        self,
        dataset_cfg: dict,
        save_path: Optional[str] = None,
    ):
        dataloader = build_dataloader(dataset_cfg)
        outputs = []
        with paddle.no_grad():
            for batch in dataloader:
                pred_dict = self.model.sample(batch)
                outputs.append(pred_dict["label"].numpy())

        result = np.concatenate(outputs, axis=0)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, prediction=result)
            logger.info(f"DeepH sampler output saved to {save_path}")
        return result
