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
try:
    from ppmat import datasets  # noqa
except Exception:
    datasets = None
try:
    from ppmat import losses  # noqa
except Exception:
    losses = None
try:
    from ppmat import metrics  # noqa
except Exception:
    metrics = None
try:
    from ppmat import models  # noqa
except Exception:
    models = None
try:
    from ppmat import optimizer  # noqa
except Exception:
    optimizer = None
try:
    from ppmat import schedulers  # noqa
except Exception:
    schedulers = None
try:
    from ppmat import trainer  # noqa
except Exception:
    trainer = None
try:
    from ppmat import utils  # noqa
except Exception:
    utils = None
try:
    from ppmat import sampler  # noqa
except Exception:
    sampler = None

__all__ = [
    "models",
    "trainer",
    "sampler",
]

try:
    # import auto-generated version information from '._version' file, using
    # setuptools_scm via 'pip install'. Details of versioning rule can be referd to:
    # https://peps.python.org/pep-0440/#public-version-identifiers
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"
