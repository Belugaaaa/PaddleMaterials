# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file adapted from https://github.com/PaddlePaddle/PaddleScience

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import paddle
from paddle import nn
from paddle import optimizer as optim
from paddle import regularizer
from paddle.incubate import optimizer as incubate_optim
from typing_extensions import Literal

from ppmat.utils import logger
from ppmat.utils import misc

if TYPE_CHECKING:
    import paddle

__all__ = ["SGD", "Momentum", "Adam", "TorchAdam", "RMSProp", "AdamW", "LBFGS", "OptimizerList"]


class SGD:
    """Stochastic Gradient Descent.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]]):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.SGD(
            learning_rate=self.learning_rate,
            parameters=parameters,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
        )
        return opt


class Momentum:
    """Simple Momentum optimizer with velocity state.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler]): The learning rate
            used to update parameter(s).
        momentum (float): Momentum factor.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]]):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
        use_nesterov (bool, optional): Whether to use nesterov momentum.
            Defaults to False.
        no_weight_decay_name (Optional[str]): List of names of no weight decay
            parameters split by white space. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler],
        momentum: float,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        use_nesterov: bool = False,
        no_weight_decay_name: Optional[str] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.use_nesterov = use_nesterov
        self.no_weight_decay_name_list = (
            no_weight_decay_name.split() if no_weight_decay_name else []
        )

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = None
        if len(self.no_weight_decay_name_list) > 0:
            params_with_decay = []
            params_without_decay = []
            for m in model_list:
                params = [
                    p
                    for n, p in m.named_parameters()
                    if not any(nd in n for nd in self.no_weight_decay_name_list)
                ]
                params_with_decay.extend(params)
                params = [
                    p
                    for n, p in m.named_parameters()
                    if any(nd in n for nd in self.no_weight_decay_name_list)
                ]
                params_without_decay.extend(params)
            parameters = [
                {"params": params_with_decay, "weight_decay": self.weight_decay},
                {"params": params_without_decay, "weight_decay": 0.0},
            ]
        else:
            parameters = (
                sum([m.parameters() for m in model_list], []) if model_list else None
            )
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            use_nesterov=self.use_nesterov,
            parameters=parameters,
        )
        if hasattr(opt, "_use_multi_tensor"):
            opt = optim.Momentum(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                grad_clip=self.grad_clip,
                parameters=parameters,
                use_nesterov=self.use_nesterov,
                use_multi_tensor=True,
            )
        return opt


class Adam:
    """Adam: A Method for Stochastic Optimization.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment
            estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment
            estimates. Defaults to 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            Defaults to 1e-08.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]])
             : Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
        lazy_mode (bool, optional): Whether to enable lazy mode for moving-average.
            Defaults to False.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        lazy_mode: bool = False,
        amsgrad: bool = False,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.lazy_mode = lazy_mode
        self.amsgrad = amsgrad

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            lazy_mode=self.lazy_mode,
            parameters=parameters,
            amsgrad=self.amsgrad,
        )
        return opt


class TorchAdam:
    """Torch-like Adam implementation used by DeepH alignment.

    This optimizer intentionally mirrors the update path already validated in the
    standalone DeepH Paddle trainer so PaddleMaterials can reproduce the same
    convergence behavior.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
    ):
        if not isinstance(learning_rate, float):
            raise NotImplementedError("TorchAdam currently expects a constant float learning rate.")
        if weight_decay is not None and not isinstance(weight_decay, (int, float)):
            raise NotImplementedError("TorchAdam currently supports only float weight_decay.")
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = 0.0 if weight_decay is None else float(weight_decay)
        self.grad_clip = grad_clip
        self.parameters = []
        self.step_id = 0
        self.state = {}

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        self.parameters = [
            p
            for m in model_list
            for p in m.parameters()
            if p.stop_gradient is False
        ]
        return self

    def clear_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.clear_gradient()

    def get_lr(self):
        return self.learning_rate

    def _clip_gradients(self):
        grads = [p.grad for p in self.parameters if p.grad is not None]
        if not grads or self.grad_clip is None:
            return

        if isinstance(self.grad_clip, nn.ClipGradByValue):
            clip_value = float(self.grad_clip.max)
            for grad in grads:
                grad.set_value(paddle.clip(grad, min=-clip_value, max=clip_value))
            return

        if isinstance(self.grad_clip, nn.ClipGradByNorm):
            clip_norm = float(self.grad_clip.clip_norm)
            for grad in grads:
                grad_norm = paddle.sqrt(paddle.sum(paddle.square(grad)))
                scale = paddle.minimum(
                    paddle.to_tensor(1.0, dtype=grad.dtype),
                    paddle.to_tensor(clip_norm, dtype=grad.dtype) / (grad_norm + 1e-12),
                )
                grad.set_value(grad * scale)
            return

        if isinstance(self.grad_clip, nn.ClipGradByGlobalNorm):
            clip_norm = float(self.grad_clip.clip_norm)
            total = None
            for grad in grads:
                value = paddle.sum(paddle.square(grad))
                total = value if total is None else total + value
            total_norm = paddle.sqrt(total)
            scale = paddle.minimum(
                paddle.to_tensor(1.0, dtype=grads[0].dtype),
                paddle.to_tensor(clip_norm, dtype=grads[0].dtype) / (total_norm + 1e-12),
            )
            for grad in grads:
                grad.set_value(grad * scale)
            return

    def step(self):
        self._clip_gradients()
        self.step_id += 1
        bias_correction1 = 1.0 - self.beta1**self.step_id
        bias_correction2 = 1.0 - self.beta2**self.step_id
        step_size = self.learning_rate / bias_correction1

        for param in self.parameters:
            grad = param.grad
            if grad is None:
                continue

            grad_data = grad
            if self.weight_decay != 0.0:
                grad_data = grad_data + self.weight_decay * param

            state = self.state.setdefault(
                param.name,
                {
                    "exp_avg": paddle.zeros_like(param),
                    "exp_avg_sq": paddle.zeros_like(param),
                },
            )
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            exp_avg_new = exp_avg * self.beta1 + grad_data * (1.0 - self.beta1)
            exp_avg_sq_new = exp_avg_sq * self.beta2 + paddle.square(grad_data) * (1.0 - self.beta2)
            denom = paddle.sqrt(exp_avg_sq_new / bias_correction2) + self.epsilon
            param_new = param - step_size * exp_avg_new / denom

            exp_avg.set_value(exp_avg_new)
            exp_avg_sq.set_value(exp_avg_sq_new)
            param.set_value(param_new)

    def state_dict(self):
        serialized_state = {}
        for name, value in self.state.items():
            serialized_state[name] = {
                "exp_avg": value["exp_avg"],
                "exp_avg_sq": value["exp_avg_sq"],
            }
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "step_id": self.step_id,
            "state": serialized_state,
        }

    def set_state_dict(self, state_dict):
        self.learning_rate = float(state_dict.get("learning_rate", self.learning_rate))
        self.beta1 = float(state_dict.get("beta1", self.beta1))
        self.beta2 = float(state_dict.get("beta2", self.beta2))
        self.epsilon = float(state_dict.get("epsilon", self.epsilon))
        self.weight_decay = float(state_dict.get("weight_decay", self.weight_decay))
        self.step_id = int(state_dict.get("step_id", 0))

        parameter_map = {param.name: param for param in self.parameters}
        self.state = {}
        for name, value in state_dict.get("state", {}).items():
            if name not in parameter_map:
                continue
            self.state[name] = {
                "exp_avg": paddle.clone(value["exp_avg"]),
                "exp_avg_sq": paddle.clone(value["exp_avg_sq"]),
            }


class LBFGS:
    """The L-BFGS is a quasi-Newton method for solving an unconstrained optimization
        problem over a differentiable function. Closely related is the Newton method
        for minimization.

    Args:
        learning_rate (float, optional): The learning rate
            used to update parameter(s). Defaults to 1.0.
        max_iter (int, optional): Maximal number of iterations per optimization step.
            Defaults to 1.
        max_eval (Optional[int]): Maximal number of function evaluations per
            optimization step. Defaults to None.
        tolerance_grad (float, optional): Termination tolerance on first order
            optimality. Defaults to 1e-07.
        tolerance_change (float, optional): Termination tolerance on function
            value/parameter changes. Defaults to 1e-09.
        history_size (int, optional): Update history size. Defaults to 100.
        line_search_fn (Optional[Literal["strong_wolfe"]]): Either 'strong_wolfe' or
            None. Defaults to "strong_wolfe".
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 1,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-07,
        tolerance_change: float = 1e-09,
        history_size: int = 100,
        line_search_fn: Optional[Literal["strong_wolfe"]] = "strong_wolfe",
    ):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.history_size = history_size
        self.line_search_fn = line_search_fn

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        try:
            opt = getattr(optim, "LBFGS")(
                learning_rate=self.lr,
                max_iter=self.max_iter,
                max_eval=self.max_eval,
                tolerance_grad=self.tolerance_grad,
                tolerance_change=self.tolerance_change,
                history_size=self.history_size,
                line_search_fn=self.line_search_fn,
                parameters=parameters,
            )
        except AttributeError:
            opt = getattr(incubate_optim, "LBFGS")(
                learning_rate=self.lr,
                max_iter=self.max_iter,
                max_eval=self.max_eval,
                tolerance_grad=self.tolerance_grad,
                tolerance_change=self.tolerance_change,
                history_size=self.history_size,
                line_search_fn=self.line_search_fn,
                parameters=parameters,
            )
        return opt


class RMSProp:
    """Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
        rate method.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler]): The learning rate
            used to update parameter(s)
        rho (float, optional): Factor ρ in equation. Defaults to 0.95.
        epsilon (float, optional): Factor ϵ in equation as a smoothing term.
            Defaults to 1e-6.
        momentum (float, optional):β in equation is the momentum term. Defaults to 0.0.
        weight_decay (Optional[Union[float, regularizer.L1Decay, regularizer.L2Decay]]):
            Regularization strategy. Defaults to None.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler],
        rho: float = 0.95,
        epsilon: float = 1e-6,
        momentum: float = 0.0,
        weight_decay: Optional[
            Union[float, regularizer.L1Decay, regularizer.L2Decay]
        ] = None,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )
        opt = optim.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=parameters,
        )
        return opt


class AdamW:
    """AdamW is implemented based on DECOUPLED WEIGHT DECAY REGULARIZATION.

    Args:
        learning_rate (Union[float, optim.lr.LRScheduler], optional): The learning rate
            used to update parameter(s). Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment
            estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment
            estimates. Defaults to 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            Defaults to 1e-8.
        weight_decay (float, optional): Regularization coefficient. Defaults to 0.01.
        grad_clip (Optional[Union[nn.ClipGradByNorm, nn.ClipGradByValue,
            nn.ClipGradByGlobalNorm]]): Gradient clipping strategy. Defaults to None.
        no_weight_decay_name (Optional[str]): List of names of no weight decay
            parameters split by white space. Defaults to None.
        one_dim_param_no_weight_decay (bool, optional): Apply no weight decay on
            1-D parameter(s). Defaults to False.
    """

    def __init__(
        self,
        learning_rate: Union[float, optim.lr.LRScheduler] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.001,
        grad_clip: Optional[
            Union[nn.ClipGradByNorm, nn.ClipGradByValue, nn.ClipGradByGlobalNorm]
        ] = None,
        no_weight_decay_name: Optional[str] = None,
        one_dim_param_no_weight_decay: bool = False,
        amsgrad: bool = False,
        multi_precision: bool = False,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.no_weight_decay_name_list = (
            no_weight_decay_name.split() if no_weight_decay_name else []
        )
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay
        self.amsgrad = amsgrad
        self.multi_precision = multi_precision

    def __call__(self, model_list: Union[nn.Layer, Tuple[nn.Layer, ...]]):
        # model_list is None in static graph
        if not isinstance(model_list, (tuple, list)):
            model_list = (model_list,)
        parameters = (
            sum([m.parameters() for m in model_list], []) if model_list else None
        )

        # TODO(gaotingquan): Model_list is None when in static graph, "no_weight_decay"
        # not work.
        if model_list is None:
            if (
                self.one_dim_param_no_weight_decay
                or len(self.no_weight_decay_name_list) != 0
            ):
                msg = '"AdamW" does not support setting "no_weight_decay" in static '
                +"graph. Please use dynamic graph."
                logger.error(Exception(msg))
                raise Exception(msg)

        self.no_weight_decay_param_name_list = (
            [
                p.name
                for model in model_list
                for n, p in model.named_parameters()
                if any(nd in n for nd in self.no_weight_decay_name_list)
            ]
            if model_list
            else []
        )

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += (
                [
                    p.name
                    for model in model_list
                    for n, p in model.named_parameters()
                    if len(p.shape) == 1
                ]
                if model_list
                else []
            )

        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            apply_decay_param_fun=self._apply_decay_param_fun,
            amsgrad=self.amsgrad,
            multi_precision=self.multi_precision,
        )
        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list


class OptimizerList:
    """OptimizerList which wrap more than one optimizer.
    NOTE: LBFGS is not supported yet.

    Args:
        optimizer_list (Tuple[optim.Optimizer, ...]): Optimizers listed in a tuple.
    """

    def __init__(self, optimizer_list: Tuple[optim.Optimizer, ...]):
        super().__init__()
        self._opt_list = optimizer_list
        if "LBFGS" in set(misc.typename(opt) for opt in optimizer_list):
            raise ValueError("LBFGS is not supported in OptimizerList yet.")

    def step(self):
        for opt in self._opt_list:
            opt.step()

    def clear_grad(self):
        for opt in self._opt_list:
            opt.clear_grad()

    def get_lr(self) -> float:
        """Return learning rate of first optimizer"""
        return self._opt_list[0].get_lr()

    def set_state_dict(self, state_dicts: List[Dict[str, "paddle.Tensor"]]):
        for i, opt in enumerate(self._opt_list):
            opt.set_state_dict(state_dicts[i])

    def state_dict(self) -> List[Dict[str, "paddle.Tensor"]]:
        state_dicts = [opt.state_dict() for opt in self._opt_list]
        return state_dicts

    def __len__(self) -> int:
        return len(self._opt_list)

    def __getitem__(self, idx):
        return self._opt_list[idx]

    def __setitem__(self, idx, opt):
        raise NotImplementedError("Can not modify any item in OptimizerList.")

    def __iter__(self):
        yield from iter(self._opt_list)
