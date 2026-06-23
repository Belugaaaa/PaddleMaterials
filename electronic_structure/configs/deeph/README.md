# DeepH

## Task

DeepH is integrated into the PaddleMaterials machine-learning electronic
structure (MLES) task. It predicts edge-level Hamiltonian matrix elements on
crystal graphs and complements electron-density-oriented MLES models.

## Model and Dataset

The PaddleMaterials integration contains:

- model: `ppmat/models/deeph/deeph.py`
- dataset adapter: `ppmat/datasets/deeph_dataset.py`
- collator: `ppmat/datasets/collate_fn.py::DeepHCollator`
- training config: `electronic_structure/configs/deeph/deeph_graphene.yaml`
- prediction entry: `electronic_structure/predict_deeph.py`
- sampler-style export entry: `electronic_structure/sample_deeph.py`

The dataset adapter reads DeepH processed structures and Hamiltonian labels,
reconstructs canonical crystal structures through
`BuildStructure(format="array")`, and then prepares the LCMP graph metadata used
by DeepH. The low-level Hamiltonian graph construction currently reuses the
upstream DeepH graph builder so that PaddleMaterials can reproduce the validated
DeepH physical graph semantics.

## Environment

Use an official PaddlePaddle release that is compatible with PaddleMaterials.
This integration is intended for PaddlePaddle `3.2.2` or later and does not
intentionally rely on develop-only Paddle APIs. For AI Studio reproduction, use
PaddlePaddle `3.2.2` or later and the TianShu hardware option.

Additional Python dependencies are the standard PaddleMaterials dependencies
plus the upstream DeepH data-processing package. Before using `DeepHDataset`,
make sure this import works:

```bash
python -c "import deeph"
```

## Data Preparation

The `graphene` example expects a DeepH data package with the following entry
file:

```text
./data/deeph/graphene/config.ini
```

The `config.ini` file should point to the DeepH processed raw data directory and
graph cache directory. The processed structure folders are expected to contain
DeepH files such as:

```text
lat.dat
site_positions.dat
element.dat
rc.npz
```

For this PR, dataset files, pretrained weights, logs, and alignment evidence are
being provided to reviewers for official BCE link backfill before merge.

Dataset split follows the DeepH config ratios used by the graphene baseline:

- train ratio: `0.6`
- validation ratio: `0.2`
- test ratio: `0.2`
- split seed: `42`

## Training

Run from the PaddleMaterials repository root:

```bash
python electronic_structure/train.py \
  -c electronic_structure/configs/deeph/deeph_graphene.yaml
```

Key training settings in `deeph_graphene.yaml`:

- epochs: `5`
- batch size: `4`
- optimizer: `TorchAdam`
- learning rate: `0.001`
- gradient clip norm: `4.2`
- target: Hamiltonian matrix elements

## Evaluation

The standard trainer runs evaluation and test phases when `Global.do_eval` and
`Global.do_test` are enabled in the config:

```yaml
Global:
  do_train: True
  do_eval: True
  do_test: True
```

Run the same command as training to produce train/eval/test losses:

```bash
python electronic_structure/train.py \
  -c electronic_structure/configs/deeph/deeph_graphene.yaml
```

## Prediction

After preparing data and a checkpoint, run:

```bash
python electronic_structure/predict_deeph.py \
  --config electronic_structure/configs/deeph/deeph_graphene.yaml \
  --checkpoint path/to/best.pdparams \
  --split test \
  --save-path output/deeph_predictions.npz \
  --summary-path output/deeph_prediction_summary.json
```

For sampler-style prediction export:

```bash
python electronic_structure/sample_deeph.py \
  --config electronic_structure/configs/deeph/deeph_graphene.yaml \
  --checkpoint path/to/best.pdparams \
  --split test \
  --save-path output/deeph_samples.npz
```

DeepH is a supervised Hamiltonian-regression model, so this sampler entry exports
model predictions rather than generating new crystal structures.

## Reference Results

### Forward Alignment

| Dataset | max_abs_diff | mean_abs_diff |
| --- | ---: | ---: |
| graphene | `1.1444e-05` | `2.3287e-07` |
| TBG_subset | `1.3351e-05` | `2.9067e-07` |

### Two-step Training Alignment

| Dataset | Step 0 loss_diff | Step 1 loss_diff |
| --- | ---: | ---: |
| graphene | `0.0` | `4.0531e-06` |
| TBG_subset | `5.9605e-08` | `1.6809e-05` |

### Supervised Metric Alignment on Graphene

| Framework | train_loss | val_loss | test_loss |
| --- | ---: | ---: | ---: |
| Torch | `0.01066117` | `0.01061319` | `0.01061640` |
| Paddle | `0.01077699` | `0.01072210` | `0.01072441` |

Absolute `test_loss` diff: `1.0801e-04`.

### Compiler Eval Benchmark

| Mode | Avg latency |
| --- | ---: |
| Paddle dynamic | `21.6952 ms` |
| Paddle to_static/CINN | `11.4824 ms` |

Speedup: `88.9434%`.

## References

- DeepH project: https://github.com/mzjb/DeepH-pack
- PaddleMaterials PR: https://github.com/PaddlePaddle/PaddleMaterials/pull/289

## Notes

- ImageNet accuracy is not applicable to this electronic-structure regression
  task.
- Generative sampling metrics are not applicable because DeepH is not a
  generative model.
